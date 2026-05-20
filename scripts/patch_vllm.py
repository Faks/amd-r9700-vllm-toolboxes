#!/usr/bin/env python3
"""
patch_vllm.py  –  Build-time patches for AMD Radeon R9700 (gfx1201)

Run this script from the root of a freshly cloned vLLM repository.
It is called by both Dockerfile and Dockerfile.rocm7.2.1 via:

    COPY scripts/patch_vllm.py patch_vllm.py
    RUN python3 patch_vllm.py

Every patch is idempotent: re-running the script on an already-patched
tree is safe (it will skip patches that are already applied).
"""

import os
import re
import sys
from pathlib import Path

_OK = 0
_WARN = 0


def _patch(path_str: str, description: str, fn):
    """Apply *fn* to the text of *path_str*.  Handles missing files and
    tracks warnings globally."""
    global _OK, _WARN
    p = Path(path_str)
    if not p.exists():
        print(f"  SKIP  {path_str}  (file not found)")
        _WARN += 1
        return
    original = p.read_text()
    result = fn(original)
    if result is None or result == original:
        print(f"  NOOP  {path_str}  ({description} – already applied or no match)")
        _OK += 1
        return
    p.write_text(result)
    print(f"  OK    {path_str}  ({description})")
    _OK += 1


# ── Patch helpers ─────────────────────────────────────────────────────


def patch_1_init_py(txt: str) -> str:
    """Force is_rocm=True and bypass all amdsmi checks."""
    txt = txt.replace("import amdsmi", "# import amdsmi")
    txt = re.sub(r"is_rocm = .*", "is_rocm = True", txt)
    txt = re.sub(
        r"if len\(amdsmi\.amdsmi_get_processor_handles\(\)\) > 0:", "if True:", txt
    )
    txt = txt.replace("amdsmi.amdsmi_init()", "pass")
    txt = txt.replace("amdsmi.amdsmi_shut_down()", "pass")
    return txt


def patch_2_rocm_py_mock(txt: str) -> str:
    """Mock amdsmi module and set device name/type."""
    # Add mock header (once)
    header = (
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        'sys.modules["amdsmi"] = MagicMock()\n'
    )
    if 'sys.modules["amdsmi"]' not in txt:
        txt = header + txt

    # Force device_type and device_name attributes
    txt = re.sub(r'device_type\s*:\s*str\s*=\s*"[^"]*"', 'device_type: str = "rocm"', txt)
    txt = re.sub(r'device_name\s*:\s*str\s*=\s*"[^"]*"', 'device_name: str = "gfx1201"', txt)

    # Override get_device_name — MUST return "AMD-gfx1201" for AITER JIT
    if "AMD-gfx1201" not in txt:
        txt += (
            "\n"
            "    @classmethod\n"
            "    def get_device_name(cls, device_id: int = 0) -> str:\n"
            '        return "AMD-gfx1201"\n'
        )
    return txt


def patch_3_transformers_config(txt: str) -> str:
    """Fix GenerationConfig lazy load imports for compatibility."""
    txt = txt.replace(
        "from transformers import GenerationConfig, PretrainedConfig",
        "from transformers.generation import GenerationConfig\n"
        "from transformers.configuration_utils import PretrainedConfig",
    )
    txt = txt.replace(
        "from transformers import PretrainedConfig, GenerationConfig",
        "from transformers.generation import GenerationConfig\n"
        "from transformers.configuration_utils import PretrainedConfig",
    )
    txt = txt.replace(
        "from transformers import PretrainedConfig",
        "from transformers.configuration_utils import PretrainedConfig",
    )
    txt = txt.replace(
        "from transformers import GenerationConfig",
        "from transformers.generation import GenerationConfig",
    )
    return txt


def patch_4_gcn_arch(txt: str) -> str:
    """Hardcode _GCN_ARCH to bypass MagicMock regex crash."""
    return re.sub(r"_GCN_ARCH\s*=\s*_get_gcn_arch\(\)", '_GCN_ARCH = "gfx1201"', txt)


def patch_5_spinloop(txt: str) -> str:
    """Fix mwaitxintrin.h include for ROCm Clang 23+."""
    return txt.replace("#include <mwaitxintrin.h>", "#include <x86intrin.h>")


def patch_6_on_mi3xx(txt: str) -> str:
    """Add gfx1201 to _ON_MI3XX gate to unlock AITER and FP8 Triton paths."""
    return txt.replace(
        '_ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942", "gfx950"])',
        '_ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942", "gfx950", "gfx1201"])',
    )


def patch_7_aiter_ops(txt: str) -> str:
    """Map gfx1201 to MI350X in AITER arch detection."""
    marker = "map gfx1201 to MI350X"
    if marker in txt:
        return txt  # already applied

    replacement = (
        "IS_AITER_FOUND = is_aiter_found()\n"
        "\n"
        "# R9700/RDNA4: map gfx1201 to MI350X in AITER arch detection\n"
        "if IS_AITER_FOUND:\n"
        "    try:\n"
        "        import aiter.ops.triton.utils.arch_info as _arch_info\n"
        '        _arch_info._ARCH_TO_DEVICE["gfx1201"] = "MI350X"\n'
        "    except (ImportError, AttributeError):\n"
        "        pass"
    )
    return txt.replace("IS_AITER_FOUND = is_aiter_found()", replacement)


def patch_8_fp8_utils(txt: str) -> str:
    """Fall back to AMD_Instinct_MI300X configs for FP8 block-scaled GEMM
    when no device-specific config file exists on gfx12 hardware."""
    marker = 'fallback_device_name = "AMD_Instinct_MI300X"'
    if marker in txt:
        return txt  # already applied

    # The target block we're extending — the config_file_path assignment
    # followed by the os.path.exists check.
    target = (
        "    config_file_path = os.path.join(\n"
        '        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name\n'
        "    )"
    )
    replacement = (
        "    config_file_path = os.path.join(\n"
        '        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name\n'
        "    )\n"
        "    # R9700/RDNA4: fall back to MI300X tuning configs\n"
        '    if not os.path.exists(config_file_path) and ("gfx12" in device_name or "Radeon" in device_name or "Graphics" in device_name):\n'
        '        fallback_device_name = "AMD_Instinct_MI300X"\n'
        "        fallback_json = f\"N={N},K={K},device_name={fallback_device_name},dtype=fp8_w8a8,block_shape=[{block_n},{block_k}].json\"\n"
        "        fallback_path = os.path.join(\n"
        '            os.path.dirname(os.path.realpath(__file__)), "configs", fallback_json\n'
        "        )\n"
        "        if os.path.exists(fallback_path):\n"
        "            config_file_path = fallback_path"
    )
    return txt.replace(target, replacement)


def patch_9_int8_utils(txt: str) -> str:
    """Fall back to AMD_Instinct_MI300X configs for INT8 block-scaled GEMM
    when no device-specific config file exists on gfx12 hardware."""
    marker = 'fallback_device_name = "AMD_Instinct_MI300X"'
    if marker in txt:
        return txt  # already applied

    # Note: int8_utils uses "block_shape=[{block_n}, {block_k}]" (with space)
    target = (
        "    config_file_path = os.path.join(\n"
        '        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name\n'
        "    )"
    )
    replacement = (
        "    config_file_path = os.path.join(\n"
        '        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name\n'
        "    )\n"
        "    # R9700/RDNA4: fall back to MI300X tuning configs\n"
        '    if not os.path.exists(config_file_path) and ("gfx12" in device_name or "Radeon" in device_name or "Graphics" in device_name):\n'
        '        fallback_device_name = "AMD_Instinct_MI300X"\n'
        "        fallback_json = f\"N={N},K={K},device_name={fallback_device_name},dtype=int8_w8a8,block_shape=[{block_n}, {block_k}].json\"\n"
        "        fallback_path = os.path.join(\n"
        '            os.path.dirname(os.path.realpath(__file__)), "configs", fallback_json\n'
        "        )\n"
        "        if os.path.exists(fallback_path):\n"
        "            config_file_path = fallback_path"
    )
    return txt.replace(target, replacement)


# ── Main ──────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("patch_vllm.py — R9700 (gfx1201) build-time patches")
    print("=" * 60)

    # Patch 1: __init__.py
    _patch("vllm/platforms/__init__.py", "force ROCm, bypass amdsmi", patch_1_init_py)

    # Patches 2, 4, 6 all target rocm.py — apply in sequence on one read
    rocm_path = "vllm/platforms/rocm.py"
    def patch_rocm_combined(txt):
        txt = patch_2_rocm_py_mock(txt)
        txt = patch_4_gcn_arch(txt)
        txt = patch_6_on_mi3xx(txt)
        return txt
    _patch(rocm_path, "mock amdsmi + GCN arch + MI3XX gate", patch_rocm_combined)

    # Patch 3: transformers config imports
    _patch(
        "vllm/transformers_utils/config.py",
        "fix GenerationConfig imports",
        patch_3_transformers_config,
    )

    # Patch 5: spinloop.cpp header
    _patch("csrc/spinloop.cpp", "fix mwaitxintrin.h for ROCm Clang", patch_5_spinloop)

    # Patch 7: AITER arch mapping
    _patch("vllm/_aiter_ops.py", "map gfx1201 to MI350X", patch_7_aiter_ops)

    # Patch 8: FP8 config fallback
    _patch(
        "vllm/model_executor/layers/quantization/utils/fp8_utils.py",
        "MI300X FP8 config fallback",
        patch_8_fp8_utils,
    )

    # Patch 9: INT8 config fallback
    _patch(
        "vllm/model_executor/layers/quantization/utils/int8_utils.py",
        "MI300X INT8 config fallback",
        patch_9_int8_utils,
    )

    print()
    print(f"Done.  {_OK} patches processed, {_WARN} warnings.")
    if _WARN > 0:
        print("(Warnings are expected if some files are absent in this vLLM version)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
