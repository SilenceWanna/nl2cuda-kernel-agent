#!/usr/bin/env bash
# autotest.sh —— 阶段7 自动探测"当前在做的 case"并调 run_on_a100.sh --strict。
#
# 用途：作为 aider `--test-cmd` 的固定值（case 名动态、不写死），实现真正零配置的 --auto-test 全自主。
# 探测顺序：①cases/ 下有未提交改动（含未跟踪）的 case；②否则取最近修改的 case。
# 找到唯一 case 后跑 `run_on_a100.sh <case> --gpu <GPU> --strict [--round-cap N]`，透传其退出码（PASS=0/否则1）。
#
# 用法：bash skill/scripts/autotest.sh            # 自动探测 case
#       GPU=5 ROUND_CAP=12 bash skill/scripts/autotest.sh   # 用 env 调 GPU/轮次上限
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
GPU="${GPU:-7}"
ROUND_CAP="${ROUND_CAP:-12}"

cd "$REPO" || { echo "VERDICT=SSH_ERROR (repo not found)"; exit 1; }

# 探测"当前在做的 case"。优先按**最近修改时间**（mtime，不受 CRLF 影响）——
# 跨文件系统（Windows↔WSL /mnt）时 git status 会把纯行尾差异误报为全 case 改动，故不用 git status。
# 取 cases/<name>/ 下最近被改过的文件所属的那个 case。
NEWEST="$(find cases -type f \( -name '*.cu' -o -name '*.py' \) -not -path '*/delivery/*' \
  -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | sed -n 's#.* cases/\([^/]*\)/.*#\1#p')"

CASE="$NEWEST"
if [ -z "$CASE" ]; then
  echo "autotest: 未找到任何 case（cases/ 下无 .cu/.py）" >&2
  echo "VERDICT=SSH_ERROR (no case detected)"; exit 1
fi

echo "[autotest] 目标 case=$CASE（按最近修改时间探测）  gpu=$GPU  round-cap=$ROUND_CAP" >&2
exec bash "$SCRIPT_DIR/run_on_a100.sh" "$CASE" --gpu "$GPU" --strict --round-cap "$ROUND_CAP"
