"""
Centralized model execution profiles for R9700 benchmark and launcher.
"""

GPU_UTIL = "0.95"
OFF_NUM_PROMPTS = 500 
OFF_FORCED_OUTPUT = "512"
DEFAULT_BATCH_TOKENS = "8192"

MODEL_TABLE = {
    # 1. Llama 3.1 8B Instruct
    # MI50 VRAM budget: 32GB per GPU. With TP=2, model weights ~7.6 GiB + KV cache.
    # Reduced from 64 seqs / 16K ctx to avoid MLP activation OOM (214 MiB gate_up_proj).
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "trust_remote": False,
        "valid_tp": [1,2],
        "max_num_seqs": "64",
        "max_tokens": "32768",
        "ctx": "65536",
        "extra_flags": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "llama3_json",
        ]
    },

    # 2. Qwen 3.5 9B (Native FP16)
    "Qwen/Qwen3.5-9B": {
        "trust_remote": True,
        "valid_tp": [1,2],
        "max_num_seqs": "64",
        "max_tokens": "32768",
        "ctx": "65536",
        "language_model_only": True,
        "extra_flags": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--reasoning-parser", "qwen3",
        ]
    },

    # 3. Qwen 3.5 27B (Native FP16) — tight fit on 1x32GB
    # which was eating the remaining VRAM headroom during warmup.
    "cyankiwi/Qwen3.6-27B-AWQ-INT4": {
        "trust_remote": True,
        "valid_tp": [1,2],
        "max_num_seqs": "32",
        "max_tokens": "16384",
        "ctx": "20480",
        "language_model_only": True,
        "enforce_eager": True,
        "gpu_util": "0.95",
        "extra_flags": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--reasoning-parser", "qwen3",
        ]
    },

    # 4. Qwen 3.5 35B AWQ (VL Model forced to Language Only)
    "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit": {
        "trust_remote": True,
        "valid_tp": [1,2],
        "max_num_seqs": "32",
        "max_tokens": "16384",
        "ctx": "20480",
        "language_model_only": True,
        "enforce_eager": True,
        "extra_flags": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--reasoning-parser", "qwen3",
        ]
    },

    "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit": {
        "trust_remote": True,
        "valid_tp": [1,2],
        "max_num_seqs": "32",
        "max_tokens": "2048",
        "ctx": "8192",
        "language_model_only": True,
        "enforce_eager": True,
        "gpu_util": "0.90",
        "kv_cache_dtype": "fp8",
        "extra_flags": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "gemma4",
            "--reasoning-parser", "gemma4",
        ]
    },

    "cyankiwi/gemma-4-31B-it-AWQ-4bit": {
        "trust_remote": True,
        "valid_tp": [1,2],
        "max_num_seqs": "32",
        "max_tokens": "2048",
        "ctx": "4096",
        "language_model_only": True,
        "enforce_eager": True,
        "gpu_util": "0.90",
        "kv_cache_dtype": "fp8",
        "extra_flags": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "gemma4",
            "--reasoning-parser", "gemma4",
        ]
    },

    "RedHatAI/Qwen3.6-35B-A3B-FP8": {
        "trust_remote": True,
        "valid_tp": [2],
        "max_num_seqs": "32",
        "max_tokens": "2048",
        "ctx": "4096",
        "language_model_only": True,
        "enforce_eager": True,
        "gpu_util": "0.90",
        "kv_cache_dtype": "fp8",
        "extra_flags": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--reasoning-parser", "qwen3",
        ]
    },

    "RedHatAI/Qwen3.6-27B-FP8": {
        "trust_remote": True,
        "valid_tp": [2],
        "max_num_seqs": "32",
        "max_tokens": "16384",
        "ctx": "32768",
        "language_model_only": True,
        "gpu_util": "0.95",
        "extra_flags": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--reasoning-parser", "qwen3",
        ]
    }
}

MODELS_TO_RUN = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen3.5-9B",
    "cyankiwi/Qwen3.6-27B-AWQ-INT4",
    "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
    "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
    "cyankiwi/gemma-4-31B-it-AWQ-4bit",
    "RedHatAI/Qwen3.6-35B-A3B-FP8",
    "RedHatAI/Qwen3.6-27B-FP8"
]
