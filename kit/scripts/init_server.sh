#!/bin/bash
# 新服务器硬件自动侦测脚本：生成当前服务器可用的 profiles.yaml
# 用法：
#   bash scripts/init_server.sh                  # 写入 <KIT_DIR>/profiles.yaml（需写权限）
#   bash scripts/init_server.sh ~/.config/sscha_kit/profiles.yaml  # 普通用户写个人配置
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KIT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_FILE="${1:-$KIT_DIR/profiles.yaml}"
mkdir -p "$(dirname "$OUT_FILE")"

echo "=== 1. 探测作业调度器 ==="
SCHED="unknown"
if command -v sbatch >/dev/null 2>&1 && command -v sinfo >/dev/null 2>&1; then
    SCHED="slurm"
elif command -v qsub >/dev/null 2>&1; then
    SCHED="pbs"
fi
echo "检测到调度器: $SCHED"

echo "=== 2. 探测 pw.x 与编译环境 ==="
PW_BIN="$(command -v pw.x || true)"
ENV_SETUP=""
if [ -z "$PW_BIN" ]; then
    PW_BIN="$(for cand in /public/apps/qe*/bin/pw.x /public/software/qe*/bin/pw.x /opt/qe*/bin/pw.x; do [ -x "$cand" ] && echo "$cand"; done | grep -v -i debug | sort -V | tail -n 1 || true)"
fi
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    ENV_SETUP="source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1"
fi
if [ -n "$PW_BIN" ]; then
    PW_DIR="$(dirname "$PW_BIN")"
    echo "检测到 pw.x: $PW_BIN"
    if [ -n "$ENV_SETUP" ]; then
        ENV_SETUP=$(printf "%s\n    export PATH=%s:\$PATH" "$ENV_SETUP" "$PW_DIR")
    else
        ENV_SETUP="export PATH=${PW_DIR}:\$PATH"
    fi
else
    echo "WARNING: 未自动找到 pw.x，请在生成的 $OUT_FILE 中手动补充 env_setup"
    ENV_SETUP="# 请填写 module load qe 或 export PATH=/path/to/qe/bin:\$PATH"
fi

echo "=== 3. 探测队列与单节点核数，生成 $OUT_FILE ==="
{
    echo "# 由 init_server.sh 在 $(hostname) ($(date '+%F %T')) 自动探测生成"
    echo "# 节点数请在各自项目 job.yaml 里用 max_nodes 与 nodes_per_job 控制。"
    echo ""
    if [ "$SCHED" = "pbs" ] && command -v pbsnodes >/dev/null 2>&1; then
        pbsnodes -a | awk '
        /^[^ ]/ { if (n != "" && prop != "") count[prop "," np]++; n=$1; np=0; prop="" }
        /np =/ { np=$3 }
        /properties =/ { prop=$3 }
        END {
            if (n != "" && prop != "") count[prop "," np]++
            for (k in count) {
                split(k, a, ",")
                q=a[1]; c=a[2]
                if (!(q in best_c) || c+0 < best_c[q]+0) best_c[q]=c
            }
            for (q in best_c) {
                printf "%s %s\n", q, best_c[q]
            }
        }' | sort | while read -r q np; do
            [ -z "$q" ] && continue
            npool=4
            if [ $((np % 4)) -ne 0 ]; then
                npool=2
                [ $((np % 2)) -ne 0 ] && npool=1
            fi
            cat <<EOF
auto_${q}_${np}:
  scheduler: pbs
  queue: ${q}
  n_nodes: 1
  n_cpu: ${np}
  n_pool: ${npool}
  batch_size: 4
  mpi_cmd: "mpirun -hostfile \$PBS_NODEFILE -np NPROC"
  env_setup: |
    ${ENV_SETUP}

EOF
        done
    elif [ "$SCHED" = "slurm" ]; then
        sinfo -h -o "%P %c" | sed 's/\*//' | awk '!seen[$1]++ {print $1, $2}' | while read -r part cpus; do
            [ -z "$part" ] && continue
            npool=4
            if [ $((cpus % 4)) -ne 0 ]; then
                npool=2
                [ $((cpus % 2)) -ne 0 ] && npool=1
            fi
            cat <<EOF
auto_${part}_${cpus}:
  scheduler: slurm
  queue: ${part}
  n_nodes: 1
  n_cpu: ${cpus}
  n_pool: ${npool}
  batch_size: 4
  mpi_cmd: "srun --mpi=pmi2 -n NPROC"
  env_setup: |
    ${ENV_SETUP}

EOF
        done
    else
        echo "WARNING: 未能自动扫描队列，写入默认模板。" >&2
        cat "$KIT_DIR/profiles.yaml"
    fi
} > "$OUT_FILE"

echo "已生成: $OUT_FILE"
# 自动为节点数最多（或首个非 debug）的主力队列添加 default 档位，支持 job.yaml 写 profile: auto
python3 - "$OUT_FILE" <<'PYDEF' 2>/dev/null || true
import sys
path = sys.argv[1]
txt = open(path, encoding="utf-8").read()
if "\ndefault:\n" not in txt:
    # Pick first profile block or batch3/cpu if present
    lines = txt.splitlines()
    blocks = []
    cur = []
    for ln in lines:
        if ln and not ln.startswith(" ") and not ln.startswith("#") and ln.endswith(":"):
            if cur:
                blocks.append(cur)
            cur = [ln]
        elif cur:
            if ln.strip() == "" and len(cur) > 3:
                blocks.append(cur)
                cur = []
            else:
                cur.append(ln)
    if cur:
        blocks.append(cur)
    if blocks:
        chosen = blocks[0]
        for b in blocks:
            hdr = b[0].lower()
            if any(k in hdr for k in ("batch3", "cpu", "compute", "normal", "work")):
                chosen = b
                break
        def_block = ["default:"] + chosen[1:]
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n# 自动选定的本机默认队列（供 job.yaml 中 profile: auto 使用）\n")
            f.write("\n".join(def_block) + "\n")
PYDEF
grep -E "^[a-zA-Z0-9_]+:$|queue:|n_cpu:|n_pool:" "$OUT_FILE" | head -n 35 || true
