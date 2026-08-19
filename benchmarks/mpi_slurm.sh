#!/usr/bin/env bash
#SBATCH --job-name=pepsy-mpi
#SBATCH --output=pepsy-mpi-%j.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2

set -euo pipefail

# Submit this file with `sbatch benchmarks/mpi_slurm.sh`. Configure the Python
# environment in the batch job or module system before invoking the script.
# Override the defaults with, for example, `srun_args="--mpi=pmix"`.
srun_args="${srun_args:---mpi=${SLURM_MPI_TYPE:-pmix}}"
exec srun ${srun_args} python benchmarks/mpi_shots.py \
  --shots "${PEPSY_MPI_SHOTS:-10000}" \
  --qubits "${PEPSY_MPI_QUBITS:-16}" \
  --depth "${PEPSY_MPI_DEPTH:-8}" \
  --chi "${PEPSY_MPI_CHI:-64}" \
  --workers "${PEPSY_MPI_WORKERS:-auto}" \
  --strategy "${PEPSY_MPI_STRATEGY:-independent}" \
  "$@"
