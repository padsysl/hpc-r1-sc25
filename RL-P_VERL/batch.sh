#!/usr/bin/env bash

#SBATCH --job-name=hpcr1-rl-demo
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1  # CRUCIAL! Only "1 task per dist per node"!
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=4
#SBATCH --constraint=gpu&hbm80g
#SBATCH --gpu-bind=none   # old perlmutter script used 'none'
#SBATCH --account=m4141_g
#SBATCH --gpu-bind=closest
#SBATCH --network=job_vni 
#SBATCH --time=0-00:14:59
#SBATCH --output=./logs/slurm.%x.%j.out

set -e -o pipefail

# Note start of script
echo "##### JOB STARTED #####"

# Load Environment
source sourceme.sh

# Get number of GPUs per node
set -x
GPUS_PER_NODE="$(nvidia-smi -L | wc -l)"
set +x
export GPUS_PER_NODE
echo "[DEBUG] GPUS_PER_NODE: $GPUS_PER_NODE"

# Get first node information
set -x
FIRST_NODE_HOSTNAME=$(hostname | sed 's/\.cluster//')  # Remove the `.cluster`, etc. from the hostname
export FIRST_NODE_HOSTNAME
FIRST_NODE_IP=$(hostname --all-ip-addresses | awk '{print $2}')  # Don't use the first one! It probably won't work.
set +x
if [ -z "$FIRST_NODE_IP" ]; then
    echo "Error: Unable to determine FIRST_NODE_IP! This machine's 'hostname --all-ip-addresses': $(hostname --all-ip-addresses), resulting in 'hostname --all-ip-addresses | awk '{print $2}'' -> $(hostname --all-ip-addresses | awk '{print $2}')"
    exit 1
fi
export FIRST_NODE_IP
echo "[DEBUG] FIRST_NODE_HOSTNAME: $FIRST_NODE_HOSTNAME with IP: $FIRST_NODE_IP"

# Get world information from SLURM
set -x
NNODES=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | wc -l)
set +x
export NNODES
echo "[DEBUG] NNODES: $NNODES"
export WORLD_SIZE=$((NNODES*GPUS_PER_NODE))
# export WORLD_SIZE=2
echo "[DEBUG] WORLD_SIZE (number of GPUs): $WORLD_SIZE"

### Set up Environment ###
source .venv/bin/activate

### Launch Ray ###

# Set up Ray
export COMMS_PORT=6379
export SUBMIT_PORT=8265
export RAY_ADDRESS="http://${FIRST_NODE_IP}:${SUBMIT_PORT}"
export RAY_ADDRESS
echo "[DEBUG] RAY_ADDRESS=${RAY_ADDRESS}"

# Set Envvars
# export RAY_BACKEND_LOG_LEVEL=debug
# export RAY_LOG_TO_STDERR=1
export RAY_DEDUP_LOGS=0

# Set Torch Distributed Things
export MASTER_ADDR="${FIRST_NODE_HOSTNAME}"
export MASTER_PORT="4269"
export TORCH_DISTRIBUTED_DEBUG=INFO

# Set NCCL/RCCL Things
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV,NET
# NCCL_DEBUG_FILE="$(realpath ./logs)/slurm.${SLURM_JOB_NAME}.${SLURM_JOB_ID}.nccl_log.%h.%p.log"; export NCCL_DEBUG_FILE
export NCCL_SOCKET_IFNAME=hsn0        # NCCL/RCCL will use the high speed network to coordinate startup.

# Launch all Ray things using launcher script
# Network Options: See https://cpe.ext.hpe.com/docs/24.03/mpt/mpich/intro_mpi.html
# job_vni: Ensure that an intra-job security token is provided.
# single_node_vni: Ensures that Slingshot security tokens are provided even if the application only runs on a single host.
# def_tles=0: Prevents a limited Cassini resource that is rarely used from being exchausted by DPM.
set -x
srun --network=single_node_vni,def_tles=0 --overlap -N "${NNODES}" -n "${NNODES}" bash "$(realpath ./launch.sh)" &
set +x

# Wait for all Ray nodes to be ready
echo "[INFO] Waiting for all Ray nodes to be ready before continuing..."
sleep 120  # Note: May need to adjust this depending on how many nodes you use
echo "[INFO] Continuing to launch the job on the Ray cluster..."

# Specify where the job will be sumitted
echo "[DEBUG] RAY_ADDRESS=${RAY_ADDRESS}"

# Set up the python command that will be run
# You may also be able to use 'Qwen/Qwen3-8B' instead of a path for the model.
PY_MODULE=verl.trainer.main_ppo
PY_ARGS=(
  algorithm.adv_estimator=grpo
  data.train_files="${DATASET_PATH}/train.parquet"
  data.val_files="${DATASET_PATH}/test.parquet"
  data.train_batch_size=1024
  data.max_prompt_length=512
  data.max_response_length=1024
  data.filter_overlong_prompts=True
  data.truncation=error
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.actor.ppo_mini_batch_size=256
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32
  actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6
  actor_rollout_ref.rollout.n=5
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  algorithm.use_kl_in_reward=False
  trainer.critic_warmup=0
  # needed for vllm >= 0.8 (comment out otherwise!)
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.free_cache_engine=True
  # keep quotes here so the program receives ["console"] (not with backslashes)
  trainer.logger='["console"]'
  trainer.project_name=hpcr1_rl_demo_gsm8k
  trainer.experiment_name=qwen3_8b_function_rm
  trainer.n_gpus_per_node="${GPUS_PER_NODE}"
  trainer.nnodes="${NNODES}"
  trainer.save_freq=20
  trainer.test_freq=5
  trainer.total_epochs=15
)

# Submit the job to Ray
set -x
export PYTHONUNBUFFERED=1  # Keep them logs flowin'

RAY_RUNTIME_ENV_LOG_LEVEL=debug RAY_ADDRESS="${RAY_ADDRESS}" \
ray job submit \
  --working-dir "$(pwd)" \
  --runtime-env-json '{
    "excludes": [
      "**/.git/**",
      ".venv/**",
      "data/**",
      "**/__pycache__/**",
      "logs/**",
      ".spack-env/**",
      "models/**",
      "datasets/**",
      "**/*.safetensors",
      "**/*.pt",
      "**/*.bin"
    ]
  }' \
  -- python -m "$PY_MODULE" "${PY_ARGS[@]}"
set +x

# Shut down cluster after job completes
set -x
RAY_ADDRESS="${RAY_ADDRESS}" ray stop
set +x

# Trap and kill ray cluster if this script is going to end
trap "ray stop" EXIT
wait

echo " === Done! === "
