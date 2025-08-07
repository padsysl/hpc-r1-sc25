#!/usr/bin/env bash


#SBATCH --job-name=hpcr1-sft-demo
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1  # CRUCIAL! Only "1 task per dist per node"!
#SBATCH --cpus-per-task=32
# #SBATCH --constraint=gpu
#SBATCH --constraint=gpu&hbm80g
#SBATCH --gpus-per-task=4
# #SBATCH --gpu-bind=closest
#SBATCH --gpu-bind=none
#SBATCH --network=job_vni
#SBATCH --account=m4141_g
#SBATCH --time=0-00:29:59
#SBATCH --output=./logs/slurm.%x.%j.out

set -e -u -o pipefail

# Define important things
OUTPUT_PATH="$(pwd)/outputs/${SLURM_JOB_ID}"; export OUTPUT_PATH

# Verify that output and logs directory exist
mkdir -p ./logs
mkdir -p "${OUTPUT_PATH}"

# Load environment
source ./sourceme.sh

# Load Python venv
echo "[TRACE] Activating venv..."
source ".venv/bin/activate"
echo "[DEBUG] Python venv activated."

# Set up local versions of packages
if [ -z "${CUDA_HOME}" ]; then
    echo "[FATAL] CUDA_HOME not set!"
    exit 1
fi

if [ -z "${NCCL_HOME}" ]; then
    echo "[FATAL] NCCL_HOME not set!"
    exit 1
fi
export LD_LIBRARY_PATH="${CUDA_HOME}:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${NCCL_HOME}:${LD_LIBRARY_PATH}"

export NCCL_NET_PLUGIN="none"  # DISABLE when not using the plugin
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV,NET

# Get the hostname of first node in this job by SLURM_JOB_NODELIST
FIRST_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1) && export FIRST_NODE
echo "[DEBUG] FIRST_NODE: $FIRST_NODE"

# Get number of GPUs per node
GPUS_PER_NODE=$(srun -N 1 -n 1 nvidia-smi -L | wc -l) && export GPUS_PER_NODE
echo "[DEBUG] GPUS_PER_NODE: $GPUS_PER_NODE"

# Get other information from SLURM
NNODES=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | wc -l) && export NNODES
echo "[DEBUG] NNODES: $NNODES"
export WORLD_SIZE=$((NNODES*GPUS_PER_NODE))

export TRAIN_SPLIT="train"
export TEST_SPLIT="test"

echo "[INFO] Will use dataset path: ${DATASET_PATH}"
echo "[INFO] Will use for TRAIN split: ${TRAIN_SPLIT}"
echo "[INFO] Will use for TEST split:  ${TEST_SPLIT}"

export USING_FSDP=0  # Set this to `1` if you are using FSDP, otherwise, it should be 0

echo "[INFO] Will use accelerate config file: ${ACCELERATOR}"

# Set up Wandb
export WANDB_DISABLED=1         # (DISABLED VIA DIRECT FLAG!) On Perlmutter, the compute nodes cannot access the internet.
export WANDB_MODE=offline     # You can set it to 'offline' to make it not talk to the mothership (perlmutter should have internet acces on comp. nodes now though)

# Notify that we're starting on the more expensive stuff now
echo "######## START TIME: $(date)"
echo "######## PYTHON ENV: $(which python)"

# Copy model and dataset to a "burst buffer" (usually /tmp) to avoid file locking issues (if model and dataset are local paths)
# May also speed things up a bit if you use fast local storage like NVMe or tmpfs rather than the parallel filesystem
export USE_BURST_BUFFER=1
export BURST_BUFFER_PATH="/tmp/burst_buffer"

srun -N "${NNODES}" --tasks-per-node=1 mkdir -p "${BURST_BUFFER_PATH}"

if [ "${USE_BURST_BUFFER}" -eq 1 ]; then
    echo "[DEBUG] Using burst buffer for model and dataset..."

    srun -N "${NNODES}" --tasks-per-node=1 mkdir -p "${BURST_BUFFER_PATH}"

    if [ -d "${DATASET_PATH}" ]; then
        echo "[DEBUG] Copying dataset to /tmp because it's a local path..."
        srun -N "${NNODES}" --tasks-per-node=1 cp -r -L "${DATASET_PATH}" "${BURST_BUFFER_PATH}/"

        # Get only the name of the dataset/directory
        DATASET_NAME=$(basename "${DATASET_PATH}") && export DATASET_NAME
        export DATASET_PATH="${BURST_BUFFER_PATH}/${DATASET_NAME}"
    fi

    # Set the HF_HOME to the burst buffer path
    # Note: This is important because otherwise the HF libraries will try to create locks in ~/.cache/huggingface
    #       which will cause silent hangs when using SLURM.
    export HF_HOME="${BURST_BUFFER_PATH}"

    # Set the TRITON_CACHE_DIR to the burst buffer path
    # Note: This is important for the same reason as HF_HOME. Locks in the parallel fs are annoying.
    export TRITON_CACHE_DIR="${BURST_BUFFER_PATH}"
else
    echo "[DEBUG] Not using burst buffer for model and dataset, will not copy. Also, will not set HF_HOME (could cause issues with file locking)!"
fi

set -x

# Disable network access for libraries that may cause hangs
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
# export TQDM_DISABLE=1         # Ideally, disables all tqdm progress bars (not great when using SLURM)

# Training setup
export CUST_VERBOSITY="DEBUG"   # Custom verbosity level for own scripts
export RDZV_BACKEND=c10d
export RDZV_ID=42
export RDZV_PORT=12355
export RDZV_ENDPOINT="${FIRST_NODE}:${RDZV_PORT}"
export NCCL_SOCKET_IFNAME="=hsn0" # The = is important! See docs.

# Important Variable Parameters
export NUM_EPOCHS=150
export BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=2  # Remember to also change it in the accelerate config
export USE_LIGER="true"
export ATTN_IMPLEMENTATION="flash_attention_2"
export MAX_SEQUENCE_LENGTH="4096"
export LEARNING_RATE="2.0e-5"
export CHECKPOINT_SAVE_INTERVAL=500
export INCLUDE_TOKENS_PER_SECOND="false"  # WARNING: Enabling this can slow things down.

# Set flags for DeepSpeed/ZeRO
# NOTE: Make sure to use trailing slashes!
export DEEPSPEED_FLAGS=" \
    --gradient_checkpointing \
"

# Set Flags for FSDP
# NOTE: Make sure to use trailing slashes!
# WARNING: If using FSDP, must remove `--gradient_checkpointing`! Otherwise, will error! Need to use FSDP's activation checkpointing on its own.
export FSDP_FLAGS=" \
    --torch_dtype bfloat16 \
    --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
"

# Set flags based on what we're doing
if [ "$USING_FSDP" -eq 1 ]; then
    SPECIFIC_FLAGS="${FSDP_FLAGS}"
else
    SPECIFIC_FLAGS="${DEEPSPEED_FLAGS}"
fi
export SPECIFIC_FLAGS


# Set the python script to use
PYTHON_SCRIPT_PATH="$(realpath ./src/sft.py)"
export PYTHON_SCRIPT_PATH

# NOTE: ddp_timeout set to 90 minutes to avoid some of the issues with timeouts during long profiling write operations
# NOTE: --bf16 DOES NOT load the model in bf16. Instead, it just uses bf16 mixed precision training.
#       --torch_dtype=torch.bfloat16 should mean the model is actually loaded in bf16
export CMD=" \
    ${PYTHON_SCRIPT_PATH} \
    ${SPECIFIC_FLAGS} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${DATASET_PATH} \
    --dataset_train_split ${TRAIN_SPLIT} \
    --dataset_test_split ${TEST_SPLIT} \
    --use_liger_kernel ${USE_LIGER} \
    --attn_implementation ${ATTN_IMPLEMENTATION} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --bf16 \
    --packing \
    --max_seq_length ${MAX_SEQUENCE_LENGTH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --ddp_timeout 5400 \
    --logging_steps 5 \
    --eval_strategy epoch \
    --eval_steps 1 \
    --resume_from_checkpoint false \
    --save_strategy steps \
    --save_steps ${CHECKPOINT_SAVE_INTERVAL} \
    --include_tokens_per_second ${INCLUDE_TOKENS_PER_SECOND} \
    --output_dir ${OUTPUT_PATH} \
    --report_to none
    "

export LAUNCHER="HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file ${ACCELERATOR}  \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --num_machines ${NNODES} \
    --num_processes ${WORLD_SIZE} \
    --main_process_ip ${FIRST_NODE} \
    --main_process_port ${RDZV_PORT} \
    --machine_rank \$SLURM_PROCID \
    --rdzv_conf "rdzv_backend=${RDZV_BACKEND},rdzv_endpoint=${RDZV_ENDPOINT},rdzv_id=${RDZV_ID}" \
    --max_restarts 0 \
    --role \$(hostname -s): \
    --tee 3 \
    "

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --network=single_node_vni,def_tles=0 \
    "

# Actually run the command
srun $SRUN_ARGS --jobid "${SLURM_JOB_ID}" bash -c "${LAUNCHER} --role \$SLURMD_NODENAME $CMD" 2>&1

set +x

echo "END TIME: $(date)"
