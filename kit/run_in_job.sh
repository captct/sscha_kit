#!/bin/bash
# 模式 B：驱动驻留批作业内运行(for clusters disallowing login-node daemons)
#
# PBS 用法：
#   qsub -q batch -l nodes=1:ppn=2 -l walltime=72:00:00 \
#        -v SSCHA_KIT_DIR=/public/apps/sscha_kit /public/apps/sscha_kit/run_in_job.sh
# SLURM 用法：
#   sbatch -p cpu -N 1 -n 2 -t 72:00:00 \
#        --wrap="bash /public/apps/sscha_kit/run_in_job.sh job.yaml"
KIT_DIR="${SSCHA_KIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL PYTHONPATH PYTHONHOME
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

exec "${KIT_DIR}/sscha_env/bin/python" -u "${KIT_DIR}/auto_sscha_driver.py" "${1:-job.yaml}"
