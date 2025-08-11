# Note that you may need to change on some systems (e.g., NixOS)
SHELL := /bin/bash

# --- Configuration ---
UV := uv
export CC=gcc-13
export CXX=g++-13
PYPROJECT_FILE := pyproject.toml
VERL_DIR := ./verl
VERL_COMMIT := 8fdc4d3f202f41461f4de9f42a637228e342668b
VENV_DIR := .venv
LOGS_DIR := logs
CUST_TMP_DIR := /tmp/uv-tmp-cache
DATASET_DIR := ./datasets
MODEL_DIR := ./models

# --- Parallelism Control and Environment Setup ---
# Set the max number of parallel source builds (CPU-intensive).
export UV_CONCURRENT_INSTALLS := 4
export UV_CONCURRENT_BUILDS := 4
export RAYON_NUM_THREADS := 4
# Set the max number of parallel package installations (I/O-intensive).
# Set max jobs for underlying build systems like Ninja (e.g., for PyTorch).
export MAX_JOBS := 2

export TORCH_CUDA_ARCH_LIST := 80
export NVCC_GENCODE := -gencode arch=compute_80,code=sm_80

# Fix issues with GLIBC versions on HPC
export UV_NO_BINARY_PACKAGE := flash-attn
export FLASH_ATTENTION_FORCE_BUILD := TRUE

# Computed
GSM8K_DIR := $(DATASET_DIR)/gsm8k
QWEN306_DIR := $(MODEL_DIR)/qwen3_0.6
QWEN253_DIR := $(MODEL_DIR)/qwen2.5_3

# A "stamp" file is used to track when dependencies were last installed.
# This avoids re-installing everything on every run.
INSTALL_STAMP := $(VENV_DIR)/.installed.stamp
VERL_STAMP := $(VERL_DIR)/.installed.stamp
GSM8K_STAMP := $(GSM8K_DIR)/.installed.stamp
QWEN306_STAMP := $(QWEN306_DIR)/.installed.stamp
QWEN253_STAMP := $(QWEN253_DIR)/.installed.stamp

# --- Core Rules ---

.DEFAULT_GOAL := all

.PHONY: all datasets models clean

.DELETE_ON_ERROR:

all: $(INSTALL_STAMP) $(VERL_STAMP) datasets models $(LOGS_DIR)
	@echo "✅ Environment is up to date."

$(INSTALL_STAMP): $(PYPROJECT_FILE)
	@echo "--> Syncing Python environment with $(PYPROJECT_FILE)..."
	@$(UV) sync --all-groups --all-extras --refresh --cache-dir="$(CUST_TMP_DIR)" --compile-bytecode
	@mkdir -p $(VENV_DIR)
	@touch $@

$(LOGS_DIR):
	@echo "--> Creating logs directory..."
	@mkdir -p $@

# Download the desired version of VERL and set it up
$(VERL_DIR):
	@git clone https://github.com/volcengine/verl.git "$(VERL_DIR)"

$(VERL_STAMP): $(VERL_DIR) $(INSTALL_STAMP)
	@git -C "$(VERL_DIR)" checkout $(VERL_COMMIT)
	@$(UV) pip install -e "$(VERL_DIR)" --compile-bytecode
	@touch $@

# Download the desired datasets
$(DATASET_DIR):
	@echo "--> Downloading and setting up datasets..."
	@mkdir -p $@

$(GSM8K_DIR): $(DATASET_DIR)
	@git lfs install
	@git clone https://huggingface.co/datasets/openai/gsm8k $@

$(GSM8K_STAMP): $(GSM8K_DIR)
	@# @mv $(GSM8K_DIR)/main/test-00000-of-00001.parquet $(GSM8K_DIR)/main/test.parquet
	@# @mv $(GSM8K_DIR)/main/train-00000-of-00001.parquet $(GSM8K_DIR)/main/train.parquet
	@# @$(UV) run python -c 'import pyarrow.parquet as pq, pyarrow as pa; t=pq.read_table("$(GSM8K_DIR)/main/test-00000-of-00001.parquet"); t=t.rename_columns(["prompt" if c=="question" else "new2" if c=="old2" else c for c in t.column_names]); pq.write_table(t,"$(GSM8K_DIR)/main/test.parquet")'
	@# @$(UV) run python -c 'import pyarrow.parquet as pq, pyarrow as pa; t=pq.read_table("$(GSM8K_DIR)/main/train-00000-of-00001.parquet"); t=t.rename_columns(["prompt" if c=="question" else "new2" if c=="old2" else c for c in t.column_names]); pq.write_table(t,"$(GSM8K_DIR)/main/train.parquet")'
	@# @rm $(GSM8K_DIR)/main/test-00000-of-00001.parquet
	@# @rm $(GSM8K_DIR)/main/train-00000-of-00001.parquet
	@$(UV) run python ./src/preprocess_gsm8k.py --input-dir="$(GSM8K_DIR)" --output-dir="$(GSM8K_DIR)/processed"
	@touch $@

datasets: $(GSM8K_STAMP)
	@echo "↳ Downloaded and set up datasets."

# Download and set up desired models
$(MODEL_DIR):
	@echo "--> Downloading and setting up models..."
	@mkdir -p $@

$(QWEN306_DIR): $(MODEL_DIR)
	@git lfs install
	@git clone https://huggingface.co/Qwen/Qwen3-0.6B $@

$(QWEN306_STAMP): $(QWEN306_DIR)
	@echo "Note: No additional setup needed for Qwen3 0.6B."
	@touch $@

$(QWEN253_DIR): $(MODEL_DIR)
	@git lfs install
	@git clone https://huggingface.co/Qwen/Qwen2.5-3B $@

$(QWEN253_STAMP): $(QWEN253_DIR)
	@echo "Note: No additional setup needed for Qwen2.5 3B."
	@touch $@

models: $(QWEN306_STAMP) $(QWEN253_STAMP)
	@echo "↳ Downloaded and set up models."

# The 'clean' target removes all generated artifacts.
clean:
	@echo "--> Cleaning up generated files..."
	@# The '-f' flag prevents errors if the files don't exist.
	@# @rm -rf $(VENV_DIR) $(CUST_TMP_DIR)
	@rm -rf "$(VERL_DIR)"
	@rm -rf "$(DATASET_DIR)"
	@rm -rf "$(MODEL_DIR)"
	@rm -rf "$(LOGS_DIR)"
	-rm -rf /tmp/*
	-rm -rf /tmp/.*
	@echo "--> Removing Python cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -delete
	@echo "↳ Done cleaning up."

