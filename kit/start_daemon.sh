#!/bin/bash
# 模式 A：登录节点静默守护启动器（共享 /public/apps/sscha_kit 与个人目录通用）
# 用法：在项目目录内执行  bash /public/apps/sscha_kit/start_daemon.sh [job.yaml]
set -e
KIT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${KIT_DIR}/sscha_env/bin/python"
JOB_FILE="${1:-job.yaml}"

if [ ! -f "${JOB_FILE}" ]; then
    echo "ERROR: 当前目录下未找到任务配置文件 ${JOB_FILE}"
    exit 1
fi
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: 未找到 ${PYTHON_BIN}，请先运行 scripts/install_public.sh"
    exit 1
fi
if [ -f sscha.pid ] && kill -0 "$(cat sscha.pid)" 2>/dev/null; then
    echo "ERROR: 守护进程已在运行 (PID $(cat sscha.pid))，先 kill 再重启"
    exit 1
fi

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL PYTHONPATH PYTHONHOME
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

nohup "${PYTHON_BIN}" -u "${KIT_DIR}/auto_sscha_driver.py" "${JOB_FILE}" > sscha_daemon.log 2>&1 &
DAEMON_PID=$!
echo "${DAEMON_PID}" > sscha.pid

echo "=========================================================="
echo "SSCHA 守护进程已在后台静默启动"
echo "  套件目录     : ${KIT_DIR}"
echo "  进程 PID     : ${DAEMON_PID} (已记录至 sscha.pid)"
echo "  实时日志     : tail -f sscha_daemon.log"
echo "  收敛演化数据 : sscha_evolution*.dat"
echo "  停止守护     : kill \$(cat sscha.pid)"
echo "  清理队列作业 : qstat -u \$USER / squeue -u \$USER 辨认后定向删除"
echo "=========================================================="
