#!/bin/bash
# 一键后台启动自动化 SSCHA 计算（在当前计算目录执行即可）
set -e
KIT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
exec bash "${KIT_DIR}/start_daemon.sh" "${1:-job.yaml}"
