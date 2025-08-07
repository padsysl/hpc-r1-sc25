# RL - PADSYS VERL

## Instructions
1. Set up environment:
  ```bash
  make
  ```
2. Edit `sourcme.sh` and verify paths are correct for your environment:
  ```bash
  $EDITOR sourceme.sh
  ```
3. Change project ID in `batch.sh`:
  ```bash
  sed -i 's/m4141_g/<your id>/g' batch.sh
  ```
4. Submit job to the cluster scheduler:
  ```bash
  # Use https://github.com/adamweingram/slurmtail
  slurmtail run -nb batch.sh

  # Standard SLURM sbatch
  sbatch batch.sh
  ```
