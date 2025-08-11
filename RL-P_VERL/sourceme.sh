# Modules 
module load PrgEnv-gnu
module load cudatoolkit/12.4
module unload craype-accel-nvidia80

# Dependency Configs (make sure they align with 'Modules'!)
NVIDIA_SDK_VERSION="24.5"

export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/${NVIDIA_SDK_VERSION}/cuda/12.4"
export NCCL_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/${NVIDIA_SDK_VERSION}/comm_libs/12.4/nccl"

# Model Configuration
MODEL_PATH="$(realpath ./models/qwen2.5_3)"
export MODEL_PATH

# Dataset Configuration
DATASET_PATH="$(realpath ./datasets/gsm8k/processed)"
export DATASET_PATH
