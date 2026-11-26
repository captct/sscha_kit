#!/bin/bash
# 一键离线安装到共享目录（如 /public/apps/sscha_kit）或个人目录
# 用法（建议由 root 或具备 /public/apps 写权限的管理员执行）：
#   bash scripts/install_public.sh /public/apps/sscha_kit
set -euo pipefail

KIT_DIR="${1:-/public/apps/sscha_kit}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== 1. 安装套件脚本到 $KIT_DIR ==="
mkdir -p "$KIT_DIR/pseudos" "$KIT_DIR/scripts"
cp -a "$PKG_ROOT/kit/." "$KIT_DIR/"
cp -a "$PKG_ROOT/scripts/." "$KIT_DIR/scripts/"

echo "=== 2. 解压并重定位离线 Python 环境 (sscha_env) ==="
if [ ! -x "$KIT_DIR/sscha_env/bin/python" ]; then
    ENV_TAR=""
    for cand in "$PKG_ROOT/sscha_env.tar.gz" "$SCRIPT_DIR/sscha_env.tar.gz" "$KIT_DIR/sscha_env.tar.gz"; do
        if [ -f "$cand" ]; then
            ENV_TAR="$cand"
            break
        fi
    done
    [ -n "$ENV_TAR" ] || { echo "ERROR: 未找到 sscha_env.tar.gz"; exit 1; }
    mkdir -p "$KIT_DIR/sscha_env"
    tar -xzf "$ENV_TAR" -C "$KIT_DIR/sscha_env"
fi

set +u
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
source "$KIT_DIR/sscha_env/bin/activate"
if command -v conda-unpack >/dev/null 2>&1; then
    conda-unpack
fi
set -u

"$KIT_DIR/sscha_env/bin/python" -c "import sscha, cellconstructor, ase, yaml, matplotlib; print('SSCHA_ENV_OK')"

echo "=== 3. 探测本服务器队列并生成 profiles.yaml ==="
bash "$KIT_DIR/scripts/init_server.sh" "$KIT_DIR/profiles.yaml"

echo "=== 4. 设置全组可读可执行权限 ==="
chmod +x "$KIT_DIR/"*.sh "$KIT_DIR/"*.py "$KIT_DIR/scripts/"*.sh
chmod -R a+rX "$KIT_DIR"

echo "=========================================================="
echo "便携版 SSCHA 套件已安装至: $KIT_DIR"
echo "  - 公共赝势请放到: $KIT_DIR/pseudos/"
echo "  - 组员个人覆盖配置（可选）: ~/.config/sscha_kit/profiles.yaml"
echo "  - 组员在项目目录启动: bash $KIT_DIR/start_daemon.sh"
echo "=========================================================="
