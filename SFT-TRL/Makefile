SHELL := /bin/bash

# --- Configuration ---
UV := uv
export CC=gcc-13
export CXX=g++-13
PYPROJECT_FILE := pyproject.toml
VENV_DIR := .venv
LOGS_DIR := logs
CUST_TMP_DIR := /tmp/uv-tmp-cache

# --- Parallelism Control and Environment Setup ---
# Set the max number of parallel source builds (CPU-intensive).
export UV_CONCURRENT_INSTALLS := 8
export UV_CONCURRENT_BUILDS := $(UV_CONCURRENT_INSTALLS)
export RAYON_NUM_THREADS := $(UV_CONCURRENT_INSTALLS)
# Set the max number of parallel package installations (I/O-intensive).
# Set max jobs for underlying build systems like Ninja (e.g., for PyTorch).
export MAX_JOBS := 4

export TORCH_CUDA_ARCH_LIST := 80
export NVCC_GENCODE := -gencode arch=compute_80,code=sm_80

# Fix issues with GLIBC versions on HPC
# export UV_NO_BINARY := 1
export UV_NO_BINARY_PACKAGE := flash-attn
# export UV_NO_BUILD_PACKAGE := bitsandbytes torch torchaudio torchvision nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 triton numpy pyarrow yarl
export FLASH_ATTENTION_FORCE_BUILD := TRUE

# A "stamp" file is used to track when dependencies were last installed.
# This avoids re-installing everything on every run.
INSTALL_STAMP := $(VENV_DIR)/.installed.stamp

# --- Core Rules ---

.DEFAULT_GOAL := all

.PHONY: all clean

.DELETE_ON_ERROR:

all: $(INSTALL_STAMP) $(LOGS_DIR)
	@echo "✅ Environment is up to date."

$(INSTALL_STAMP): $(PYPROJECT_FILE)
	@echo "--> Syncing Python environment with $(PYPROJECT_FILE)..."
	@$(UV) sync --all-groups --all-extras --refresh --cache-dir="$(CUST_TMP_DIR)" --compile-bytecode
	@mkdir -p $(VENV_DIR)
	@touch $@

$(LOGS_DIR):
	@echo "--> Creating logs directory..."
	@mkdir -p $@

clean:
	@echo "--> Cleaning up generated files..."
	@rm -rf $(VENV_DIR) $(LOGS_DIR) $(CUST_TMP_DIR)
	-rm -rf /tmp/*
	-rm -rf /tmp/.*
	@echo "--> Removing Python cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -delete

