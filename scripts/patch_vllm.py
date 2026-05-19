import os
import re
from pathlib import Path

def patch_vllm():
    print("Starting vLLM patching for R9700 (gfx1201)...")

    # Patch 1: __init__.py - Force is_rocm=True and bypass amdsmi checks
    p1 = Path('vllm/platforms/__init__.py')
    if p1.exists():
        print("Patching vllm/platforms/__init__.py...")
        txt = p1.read_text()
        txt = txt.replace('import amdsmi', '# import amdsmi')
        txt = re.sub(r'is_rocm = .*', 'is_rocm = True', txt)
        txt = re.sub(r'if len\(amdsmi\.amdsmi_get_processor_handles\(\)\) > 0:', 'if True:', txt)
        txt = txt.replace('amdsmi.amdsmi_init()', 'pass')
        txt = txt.replace('amdsmi.amdsmi_shut_down()', 'pass')
        p1.write_text(txt)
    else:
        print("Warning: vllm/platforms/__init__.py not found.")

    # Patch 2: rocm.py - Mock amdsmi, set device name/type, and override get_device_name
    p2 = Path('vllm/platforms/rocm.py')
    if p2.exists():
        print("Patching vllm/platforms/rocm.py...")
        txt = p2.read_text()
        
        # Add mock header
        header = 'import sys\nfrom unittest.mock import MagicMock\nsys.modules["amdsmi"] = MagicMock()\n'
        if 'sys.modules["amdsmi"]' not in txt:
            txt = header + txt
            
        txt = re.sub(r'device_type = .*', 'device_type = "rocm"', txt)
        txt = re.sub(r'device_name = .*', 'device_name = "gfx1201"', txt)
        
        # Override get_device_name with classmethod returning 'AMD Radeon R9700'
        if 'AMD Radeon R9700' not in txt:
            txt += '\n    @classmethod\n    def get_device_name(cls, device_id: int = 0) -> str:\n        return "AMD Radeon R9700"\n'
            
        # Patch 4: rocm.py - Hardcode _GCN_ARCH to bypass MagicMock regex crash
        txt = re.sub(r'_GCN_ARCH\s*=\s*_get_gcn_arch\(\)', '_GCN_ARCH = "gfx1201"', txt)
        
        # Patch 6: rocm.py - Add gfx1201 to _ON_MI3XX to unlock AITER and FP8 Triton paths
        txt = txt.replace(
            '_ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942", "gfx950"])',
            '_ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942", "gfx950", "gfx1201"])'
        )
        p2.write_text(txt)
    else:
        print("Warning: vllm/platforms/rocm.py not found.")

    # Patch 3: transformers_utils/config.py - Fix GenerationConfig lazy load
    p3 = Path('vllm/transformers_utils/config.py')
    if p3.exists():
        print("Patching vllm/transformers_utils/config.py...")
        txt = p3.read_text()
        txt = txt.replace('from transformers import GenerationConfig, PretrainedConfig', 'from transformers.generation import GenerationConfig\nfrom transformers.configuration_utils import PretrainedConfig')
        txt = txt.replace('from transformers import PretrainedConfig, GenerationConfig', 'from transformers.generation import GenerationConfig\nfrom transformers.configuration_utils import PretrainedConfig')
        txt = txt.replace('from transformers import PretrainedConfig', 'from transformers.configuration_utils import PretrainedConfig')
        txt = txt.replace('from transformers import GenerationConfig', 'from transformers.generation import GenerationConfig')
        p3.write_text(txt)
    else:
        print("Warning: vllm/transformers_utils/config.py not found.")

    # Patch 5: csrc/spinloop.cpp - Fix mwaitxintrin.h include for ROCm Clang 23
    p5 = Path('csrc/spinloop.cpp')
    if p5.exists():
        print("Patching csrc/spinloop.cpp...")
        txt = p5.read_text()
        txt = txt.replace('#include <mwaitxintrin.h>', '#include <x86intrin.h>')
        p5.write_text(txt)
    else:
        print("Warning: csrc/spinloop.cpp not found.")

    # Patch 7: _aiter_ops.py - Add gfx1201 to AITER arch mapping to prevent KeyError crash
    p7 = Path('vllm/_aiter_ops.py')
    if p7.exists():
        print("Patching vllm/_aiter_ops.py...")
        txt = p7.read_text()
        aiter_patch = (
            'IS_AITER_FOUND = is_aiter_found()\n\n'
            '# R9700/RDNA4: map gfx1201 to MI350X in AITER arch detection\n'
            'if IS_AITER_FOUND:\n'
            '    try:\n'
            '        import aiter.ops.triton.utils.arch_info as _arch_info\n'
            '        _arch_info._ARCH_TO_DEVICE["gfx1201"] = "MI350X"\n'
            '    except (ImportError, AttributeError):\n'
            '        pass'
        )
        if 'map gfx1201 to MI350X' not in txt:
            txt = txt.replace('IS_AITER_FOUND = is_aiter_found()', aiter_patch)
        p7.write_text(txt)
    else:
        print("Warning: vllm/_aiter_ops.py not found.")

    # Patch 8: fp8_utils.py - Fall back to MI300X configs for missing Radeon/gfx12 config files
    p8 = Path('vllm/model_executor/layers/quantization/utils/fp8_utils.py')
    if p8.exists():
        print("Patching vllm/model_executor/layers/quantization/utils/fp8_utils.py...")
        txt = p8.read_text()
        target = (
            '    config_file_path = os.path.join(\n'
            '        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name\n'
            '    )'
        )
        replacement = (
            '    config_file_path = os.path.join(\n'
            '        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name\n'
            '    )\n'
            '    if not os.path.exists(config_file_path) and ("Radeon" in device_name or "gfx12" in device_name or "Graphics" in device_name):\n'
            '        fallback_device_name = "AMD_Instinct_MI300X"\n'
            '        fallback_json_file_name = f"N={N},K={K},device_name={fallback_device_name},dtype=fp8_w8a8,block_shape=[{block_n},{block_k}].json"\n'
            '        fallback_file_path = os.path.join(\n'
            '            os.path.dirname(os.path.realpath(__file__)), "configs", fallback_json_file_name\n'
            '        )\n'
            '        if os.path.exists(fallback_file_path):\n'
            '            config_file_path = fallback_file_path'
        )
        if 'fallback_device_name = "AMD_Instinct_MI300X"' not in txt:
            txt = txt.replace(target, replacement)
        p8.write_text(txt)
    else:
        print("Warning: vllm/model_executor/layers/quantization/utils/fp8_utils.py not found.")

    # Patch 9: int8_utils.py - Fall back to MI300X configs for missing Radeon/gfx12 config files
    p9 = Path('vllm/model_executor/layers/quantization/utils/int8_utils.py')
    if p9.exists():
        print("Patching vllm/model_executor/layers/quantization/utils/int8_utils.py...")
        txt = p9.read_text()
        target = (
            '    config_file_path = os.path.join(\n'
            '        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name\n'
            '    )'
        )
        replacement = (
            '    config_file_path = os.path.join(\n'
            '        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name\n'
            '    )\n'
            '    if not os.path.exists(config_file_path) and ("Radeon" in device_name or "gfx12" in device_name or "Graphics" in device_name):\n'
            '        fallback_device_name = "AMD_Instinct_MI300X"\n'
            '        fallback_json_file_name = f"N={N},K={K},device_name={fallback_device_name},dtype=int8_w8a8,block_shape=[{block_n}, {block_k}].json"\n'
            '        fallback_file_path = os.path.join(\n'
            '            os.path.dirname(os.path.realpath(__file__)), "configs", fallback_json_file_name\n'
            '        )\n'
            '        if os.path.exists(fallback_file_path):\n'
            '            config_file_path = fallback_file_path'
        )
        if 'fallback_device_name = "AMD_Instinct_MI300X"' not in txt:
            txt = txt.replace(target, replacement)
        p9.write_text(txt)
    else:
        print("Warning: vllm/model_executor/layers/quantization/utils/int8_utils.py not found.")

    print("Patches completed successfully!")

if __name__ == '__main__':
    patch_vllm()
