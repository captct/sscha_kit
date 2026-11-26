#!/bin/bash
# 一键非提交自检（在当前计算目录执行即可，不投递任何作业）
set -e
KIT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL PYTHONPATH PYTHONHOME
exec "${KIT_DIR}/sscha_env/bin/python" -s "${KIT_DIR}/verify_no_submit.py" .
