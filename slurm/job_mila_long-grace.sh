#!/bin/bash
# Author(s): Samuel Garcin (garcin.samuel@gmail.com)
#
# example usage:
# ```
# EXPT_FILE=experiments.txt  # <- this has a command to run on each line
# NR_EXPTS=`cat ${EXPT_FILE} | wc -l`
# MAX_PARALLEL_JOBS=12
# sbatch job_mila.sh $EXPT_FILE
# ```
#
# or, equivalently and as intended, with provided `run_experiement`:
# ```
# run_experiment -b slurm_arrayjob.sh -e experiments.txt -m 12
# ```


# ====================
# Options for sbatch
# ====================

# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --output=/home/mila/s/%u/idaac/slurm/slurm_logs/slurm-%A_%a.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --error=/home/mila/s/%u/idaac/slurm/slurm_logs/slurm-%A_%a.out

#SBATCH --partition=long-grace

# Maximum number of nodes to use for the job
# #SBATCH --nodes=1

# Generic resources to use - typically you'll want gpu:n to get n gpus
#SBATCH --gres=gpu:1

# Megabytes of RAM required. Check `cluster-status` for node configurations
#SBATCH --mem=25000

# Number of CPUs to use. Check `cluster-status` for node configurations
#SBATCH --cpus-per-task=6

# Maximum time for the job to run, format: days-hours:minutes:seconds
#SBATCH --time=0-16:00:00

# Requeue jobs if they fail
#SBATCH --requeue

# Exclude nodes with known issues
#SBATCH --exclude=cn-g026

#SBATCH --signal=SIGINT@120

experiment_text_file=$1
experiment_no=$2

# =====================
# Logging information
# =====================

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

# ===================
# Environment setup
# ===================

echo "Loading modules"
module load cuda/12.0
source $HOME/.bashrc

# Activate your conda environment
CONDA_ENV_NAME=idaac
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}

REPO_DIR=$HOME/idaac
export PYTHONPATH="${PYTHONPATH}:$REPO_DIR"
EXP_FILEPATH=$REPO_DIR/slurm/exp_files/${experiment_text_file}

cd $REPO_DIR
echo "Running experiment ${experiment_no} from ${experiment_text_file}"
COMMAND="srun `sed \"${experiment_no}q;d\" ${EXP_FILEPATH}`"
echo "Running provided command: ${COMMAND}"
$COMMAND
# Script is designed to output the following exit codes:
# 0: successful termination
# 1: Python error
# 2: terminated by runstate, no requeueing
# 3: terminated by runstate, needs requeueing.
# However in practice the exit code gets overwritten by other things, for example:
# 255: from other thread, even if we meant to return 2 or 3
# 134: core dumped, which is an hardware error
# Compromise: Requeue if the exit code is greater than 2, to avoid requeuing faulty code but still catch hardware errors.
# Potential issue 1: job will be requeued anyways if exit code 2 gets overwritten by an higher exit code, which can happen.
# Potential issue 2: job will not be requeued if an hardware issue causes a python error (I have not seen this happen so far)
LASTEXITCODE=$?
echo "Exited with code ${LASTEXITCODE}"

# Set limits for maximum number of restarts
SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
SLURM_RESTART_COUNT_MAX=10

# typically applications emit an exitcode
if [[ $LASTEXITCODE -gt 2 ]]; then
  if [[ $SLURM_RESTART_COUNT -lt $SLURM_RESTART_COUNT_MAX ]]; then
    echo "Requeuing current job"
    scontrol requeue $SLURM_JOB_ID
  else
    echo "SLURM_RESTART_COUNT_MAX ($SLURM_RESTART_COUNT_MAX) reached!"
  fi
fi

# ===================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
