FROM registry.fedoraproject.org/fedora:43

# 1. System Base & Build Tools
# Added 'gperftools-libs' for tcmalloc (fixes double-free)
RUN dnf -y install --setopt=install_weak_deps=False --nodocs \
  python3.12 python3.12-devel git rsync libatomic bash ca-certificates curl \
  gcc gcc-c++ binutils make ffmpeg-free \
  cmake ninja-build aria2c tar xz vim nano \
  libdrm-devel zlib-devel openssl-devel jq \
  numactl-devel gperftools-libs dialog procps-ng \
  && dnf clean all && rm -rf /var/cache/dnf/*

# 2. Install "TheRock" ROCm SDK (Tarball Method)
WORKDIR /tmp
ARG ROCM_MAJOR_VER=7
ARG GFX=gfx120X-all
RUN set -euo pipefail; \
  BASE="https://therock-nightly-tarball.s3.amazonaws.com"; \
  PREFIX="therock-dist-linux-${GFX}-${ROCM_MAJOR_VER}"; \
  KEY="$(curl -s "${BASE}?list-type=2&prefix=${PREFIX}" \
  | tr '<' '\n' \
  | grep -o "therock-dist-linux-${GFX}-${ROCM_MAJOR_VER}\..*\.tar\.gz" \
  | sort -V | tail -n1)"; \
  echo "Downloading Latest Tarball: ${KEY}"; \
  aria2c -x 16 -s 16 -j 16 --file-allocation=none "${BASE}/${KEY}" -o therock.tar.gz; \
  mkdir -p /opt/rocm; \
  tar xzf therock.tar.gz -C /opt/rocm --strip-components=1; \
  rm therock.tar.gz

# 3. Configure Global ROCm Environment
# We add LD_PRELOAD for tcmalloc here to fix the shutdown crash
RUN export ROCM_PATH=/opt/rocm && \
  BITCODE_PATH=$(find /opt/rocm -type d -name bitcode -print -quit) && \
  printf '%s\n' \
  "export ROCM_PATH=/opt/rocm" \
  "export HIP_PLATFORM=amd" \
  "export HIP_PATH=/opt/rocm" \
  "export HIP_CLANG_PATH=/opt/rocm/llvm/bin" \
  "export HIP_DEVICE_LIB_PATH=$BITCODE_PATH" \
  "export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:\$PATH" \
  "export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:\$LD_LIBRARY_PATH" \
  "export ROCBLAS_USE_HIPBLASLT=1" \
  "export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1" \
  "export VLLM_TARGET_DEVICE=rocm" \
  "export HIP_FORCE_DEV_KERNARG=1" \
  "export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1" \
  "export LD_PRELOAD=/usr/lib64/libtcmalloc_minimal.so.4" \
  > /etc/profile.d/rocm-sdk.sh && \
  chmod 0644 /etc/profile.d/rocm-sdk.sh

# 4. Python Venv Setup
RUN /usr/bin/python3.12 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONNOUSERSITE=1
RUN printf 'source /opt/venv/bin/activate\n' > /etc/profile.d/venv.sh
RUN python -m pip install --upgrade pip wheel packaging "setuptools<80.0.0"

# 5. Install PyTorch (TheRock Nightly)
RUN python -m pip install \
  --index-url https://rocm.nightlies.amd.com/v2-staging/gfx120X-all/ \
  --pre torch torchaudio torchvision && \
  find /opt/venv/lib/python3.12/site-packages/torch* -type f -name "*.so" -exec strip -s {} + 2>/dev/null || true && \
  find /opt/venv/lib/python3.12/site-packages/torch* -type d -name "__pycache__" -prune -exec rm -rf {} +

# Flash-Attention
WORKDIR /opt
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"

RUN git clone https://github.com/ROCm/flash-attention.git &&\ 
  cd flash-attention &&\
  git checkout main_perf &&\
  python setup.py install && \
  cd /opt && rm -rf /opt/flash-attention

# 6. Clone vLLM
RUN git clone https://github.com/vllm-project/vllm.git /opt/vllm
WORKDIR /opt/vllm

# --- PATCHING ---
# vLLM relies on 'amdsmi' to detect AMD GPUs. If it's missing or fails (common in containers),
# vLLM falls back to CPU. We patch it to force ROCm detection.
COPY scripts/patch_vllm.py patch_vllm.py
RUN python patch_vllm.py

# 7. Build vLLM (Wheel Method) with CLANG Host Compiler
RUN python -m pip install --upgrade cmake ninja packaging wheel numpy "setuptools-scm>=8" "setuptools<80.0.0" scikit-build-core pybind11
ENV ROCM_HOME="/opt/rocm"
ENV HIP_PATH="/opt/rocm"
ENV VLLM_TARGET_DEVICE="rocm"
ENV PYTORCH_ROCM_ARCH="gfx1201"
ENV HIP_ARCHITECTURES="gfx1201"          
ENV AMDGPU_TARGETS="gfx1201"              
ENV MAX_JOBS="4"

# --- FIX FOR SEGFAULT ---
# We force the Host Compiler (CC/CXX) to be the ROCm Clang, not Fedora GCC.
# This aligns the ABI of the compiled vLLM extensions with PyTorch.
ENV CC="/opt/rocm/llvm/bin/clang"
ENV CXX="/opt/rocm/llvm/bin/clang++"

RUN export HIP_DEVICE_LIB_PATH=$(find /opt/rocm -type d -name bitcode -print -quit) && \
  echo "Compiling with Bitcode: $HIP_DEVICE_LIB_PATH" && \
  export CMAKE_ARGS="-DROCM_PATH=/opt/rocm -DHIP_PATH=/opt/rocm -DAMDGPU_TARGETS=gfx1201 -DHIP_ARCHITECTURES=gfx1201 -DHIP_FOUND=ON" && \
  python -m pip wheel --no-build-isolation --no-deps -w /tmp/dist -v . && \
  python -m pip install /tmp/dist/*.whl && \
  rm -rf /tmp/dist && \
  find /opt/venv/lib/python3.12/site-packages/vllm -type f -name "*.so" -exec strip -s {} + 2>/dev/null || true && \
  find /opt/venv/lib/python3.12/site-packages/vllm -type d -name "__pycache__" -prune -exec rm -rf {} +

# --- bitsandbytes (ROCm) ---
WORKDIR /opt
RUN git clone -b rocm_enabled_multi_backend https://github.com/ROCm/bitsandbytes.git
WORKDIR /opt/bitsandbytes

# Explicitly set HIP_PLATFORM (Docker ENV, not /etc/profile)
ENV HIP_PLATFORM="amd"
ENV CMAKE_PREFIX_PATH="/opt/rocm"

# Force CMake to use the System ROCm Compiler (/opt/rocm/llvm/bin/clang++)
RUN cmake -S . \
  -DGPU_TARGETS="gfx1201" \
  -DBNB_ROCM_ARCH="gfx1201" \
  -DCOMPUTE_BACKEND=hip \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  && \
  make -j$(nproc) && \
  python -m pip install --no-cache-dir . --no-build-isolation --no-deps && \
  find /opt/venv/lib/python3.12/site-packages/bitsandbytes -type d -name "__pycache__" -prune -exec rm -rf {} +

# --- aiter (ROCm) ---
# Install from source to pre-compile native extensions at build time.
# Without this, aiter's JIT system tries to compile module_aiter_enum at
# runtime and fails with ModuleNotFoundError on gfx1201.
# The TheRock nightly drops a broken 'hipconfig' shim in the venv that
# depends on rocm_sdk_core (missing). Remove it so aiter falls through
# to the real /opt/rocm/bin/hipconfig.
RUN rm -f /opt/venv/bin/hipconfig && \
  python -m pip install git+https://github.com/ROCm/aiter.git && \
  chmod a+w /opt/venv/lib64/python3.12/site-packages/aiter/jit/

# 8. Runtime Configurations
WORKDIR /opt

# Install hipcc wrapper: intercepts --offload-arch=native and rewrites to gfx1201
# This is required for AITER JIT compilation and any runtime hipcc invocations.
RUN mv /opt/rocm/bin/hipcc /opt/rocm/bin/hipcc.real && \
  printf '#!/bin/bash\n# R9700 hipcc wrapper: forces single-arch gfx1201 compilation\nargs=()\nfor arg in "$@"; do\n    args+=("${arg//--offload-arch=native/--offload-arch=gfx1201}")\ndone\nexec /opt/rocm/bin/hipcc.real "${args[@]}"\n' > /opt/rocm/bin/hipcc && \
  chmod 755 /opt/rocm/bin/hipcc

COPY scripts/01-rocm-envs.sh /etc/profile.d/01-rocm-envs.sh
COPY scripts/99-toolbox-banner.sh /etc/profile.d/99-toolbox-banner.sh
COPY scripts/zz-venv-last.sh /etc/profile.d/zz-venv-last.sh
COPY scripts/start_vllm.py /usr/local/bin/start-vllm
COPY benchmarks/max_context_results.json /opt/max_context_results.json
COPY benchmarks/run_vllm_bench.py /opt/run_vllm_bench.py
COPY benchmarks/models.py /opt/models.py
RUN chmod 0644 /etc/profile.d/*.sh && chmod +x /usr/local/bin/start-vllm && chmod 0644 /opt/max_context_results.json && chmod 0644 /opt/models.py
RUN printf 'ulimit -S -c 0\n' > /etc/profile.d/90-nocoredump.sh && chmod 0644 /etc/profile.d/90-nocoredump.sh

# 9. Install Custom RCCL (gfx1201) - Replaces standard library with manually built one
COPY custom_libs/librccl.so.1.gz /tmp/librccl.so.1.gz
RUN echo "Installing Custom RCCL..." && \
  gzip -d /tmp/librccl.so.1.gz && \
  chmod 755 /tmp/librccl.so.1 && \
  # Replace /opt/rocm library strictly
  cp -fv /tmp/librccl.so.1 /opt/rocm/lib/librccl.so.1.0 && \
  # Replace /opt/venv library
  find /opt/venv -name "librccl.so*" -type f -exec cp -fv /tmp/librccl.so.1 {} + && \
  rm /tmp/librccl.so.1

CMD ["/bin/bash"]
