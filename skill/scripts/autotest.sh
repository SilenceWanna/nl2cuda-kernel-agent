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

# ① 有未提交改动（含未跟踪）的 case 目录
CANDIDATES="$(git status --porcelain -- cases/ 2>/dev/null | awk '{print $NF}' \
  | sed -n 's#^cases/\([^/]*\)/.*#\1#p' | sort -u)"

# ② 回退：最近修改的 case（按 kernels/op.py mtime）
if [ -z "$CANDIDATES" ]; then
  CANDIDATES="$(ls -1t cases/*/op.py 2>/dev/null | sed -n 's#^cases/\([^/]*\)/op.py#\1#p' | head -1)"
fi

N="$(echo "$CANDIDATES" | grep -c .)"
if [ "$N" -eq 0 ]; then
  echo "autotest: 未找到任何 case（cases/ 下无改动也无 op.py）" >&2
  echo "VERDICT=SSH_ERROR (no case detected)"; exit 1
fi
if [ "$N" -gt 1 ]; then
  echo "autotest: 探测到多个改动中的 case，无法确定目标：" >&2
  echo "$CANDIDATES" >&2
  echo "  请只改一个 case，或直接指定：bash skill/scripts/run_on_a100.sh <case> --gpu $GPU --strict" >&2
  echo "VERDICT=SSH_ERROR (ambiguous case)"; exit 1
fi

CASE="$(echo "$CANDIDATES" | tr -d '[:space:]')"
echo "[autotest] 目标 case=$CASE  gpu=$GPU  round-cap=$ROUND_CAP" >&2
exec bash "$SCRIPT_DIR/run_on_a100.sh" "$CASE" --gpu "$GPU" --strict --round-cap "$ROUND_CAP"
