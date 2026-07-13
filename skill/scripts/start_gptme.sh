#!/usr/bin/env bash
# start_gptme.sh —— 阶段7：gptme 零配置全自主启动（把 AGENTS.md 内化为 system prompt）。
#
# gptme 无标准约定文件（不像 aider 读 CONVENTIONS.md / codex 读 AGENTS.md），故用 --system
# 把本仓库的 AGENTS.md 作为 system prompt 注入——这样用户只需在 gptme 界面输入算法定义，
# gptme 就按内化的方法论自动建 case + 写实现 + 调 run_on_a100.sh 自测 + 优化到达标。
#
# 前置：WSL 需已拷密钥（cp /mnt/c/Users/<user>/.ssh/nl2cuda_gpu ~/.ssh/ && chmod 600），dongcc 代理在 8787。
# 用法：bash skill/scripts/start_gptme.sh      # 在仓库根运行
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO"

export PATH="$HOME/.local/bin:$PATH"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8787/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
export LANG=C.UTF-8 LC_ALL=C.UTF-8

SYS="$REPO/AGENTS.md"
[ -f "$SYS" ] || { echo "找不到 AGENTS.md（$SYS）" >&2; exit 1; }

echo "[start_gptme] system=AGENTS.md  提示：直接输入算法定义即可，gptme 会自动建 case+写实现+自测+优化。" >&2
exec gptme -y --model openai/GPT-5.5-joybuilder --system "$(cat "$SYS")"
