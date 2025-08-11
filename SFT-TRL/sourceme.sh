# Modules 
module load PrgEnv-gnu
module load cudatoolkit/12.4
module unload craype-accel-nvidia80

# Dependency Configs (make sure they align with 'Modules'!)
NVIDIA_SDK_VERSION="24.5"

export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/${NVIDIA_SDK_VERSION}/cuda/12.4"
export NCCL_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/${NVIDIA_SDK_VERSION}/comm_libs/12.4/nccl"
# export MPI_HOME=""

# Model Configuration
MODEL_PATH="/pscratch/sd/a/${USER}/cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
export MODEL_PATH

# Dataset Configuration
DATASET_PATH="/pscratch/sd/a/${USER}/cache/huggingface/datasets/facebook--natural_reasoning--preproc-tiny"
export DATASET_PATH

# Parallelism Configuration
ACCELERATOR="$(realpath ./accelerate_configs/perlmutter-zero-3.yaml)" && export ACCELERATOR
