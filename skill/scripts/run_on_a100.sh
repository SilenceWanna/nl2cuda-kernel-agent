#!/usr/bin/env bash
# run_on_a100.sh —— 阶段6 自主闭环的"代码搬运 + 远程自测"驱动器。
#
# 一条命令完成：打包本地 case → 经双跳 SSH 直传 A100 → 清扩展缓存 →
# 远程跑 verify（正确性门禁）→ 通过才跑 bench → 回传原始日志 + 单行机读 VERDICT。
#
# 设计要点：
#  - 只传 cases/<case>/（不传 framework/）→ A100 恒跑提交版评测基座（防作弊，结构性保证）。
#  - verify 先跑、通过才 bench（loop.md 正确性优先）；verify 失败短路为 VERDICT=VERIFY_FAIL。
#  - 走 SSH 直传而非 git（A100↔GitHub 网络不稳）；≤3 次重试抗双跳抖动。
#  - 密钥/跳板内联，不改用户 ~/.ssh/config，不进提交物之外的配置。
#  - 跑在"驱动 agent 所在环境"：Windows Git Bash（aider）或 WSL（gptme，需先把密钥拷进 WSL ~/.ssh/）。
#
# 用法：
#   bash skill/scripts/run_on_a100.sh <case> [--gpu N] [--auto-gpu] [--sync-cli]
#                                     [--workdir PATH] [--remote-repo PATH] [--size-env "K=V,..."]
#   首次运行必须带 --sync-cli（把带 --emit-verdict 的 bench_case.py 同步到 A100 一次）。
#
# 退出码：0=拿到 VERDICT（PASS 或各种 *_FAIL，具体看 VERDICT 行）；非 0=驱动器自身错误（SSH/参数）。
# VERDICT 文法：PASS · VERIFY_FAIL · BENCH_FAIL · CV_INVALID · FRAMEWORK_DIRTY · SSH_ERROR
set -uo pipefail

# ---- 双跳 SSH 目标（内联，勿写进 ~/.ssh/config）----
KEY="${NL2CUDA_KEY:-$HOME/.ssh/nl2cuda_gpu}"
JUMP="root@11.91.169.119"
GPUBOX="wangzihe@11.127.49.134"

ssh_a100() {   # 双跳 SSH 包装：$@ 为远程命令；stdin 透传（供 heredoc / tar）
  ssh -i "$KEY" -o IdentitiesOnly=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new \
      -o "ProxyCommand=ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -W %h:%p $JUMP" \
      "$GPUBOX" "$@"
}

emit() { echo "VERDICT=$1"; }   # 驱动器自身错误也用 VERDICT 表达，便于 agent 统一解析

# ---- 参数 ----
CASE=""
GPU="7"
AUTO_GPU=0
SYNC_CLI=0
STRICT=0                                      # 1=按 VERDICT 决定退出码(PASS=0/其余=1)，供 aider --auto-test 等靠退出码驱动的自主闭环
ROUND_CAP=0                                  # 0=禁用(Stage B 半自动默认)；>0 启用机械轮次上限(Stage C 全自主兜底)
REMOTE_REPO="~/nl2cuda-kernel-agent"
SIZE_ENV=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "$SCRIPT_DIR/../.." && pwd)"   # 仓库根（skill/scripts/ 的上两级）

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)         GPU="$2"; shift 2 ;;
    --auto-gpu)    AUTO_GPU=1; shift ;;
    --sync-cli)    SYNC_CLI=1; shift ;;
    --strict)      STRICT=1; shift ;;
    --workdir)     WORKDIR="$2"; shift 2 ;;
    --remote-repo) REMOTE_REPO="$2"; shift 2 ;;
    --size-env)    SIZE_ENV="$2"; shift 2 ;;
    --round-cap)   ROUND_CAP="$2"; shift 2 ;;
    -*)            echo "未知参数: $1" >&2; exit 2 ;;
    *)             CASE="$1"; shift ;;
  esac
done
[ -z "$CASE" ] && { echo "用法: run_on_a100.sh <case> [--gpu N] [--sync-cli] ..." >&2; exit 2; }
[ -d "$WORKDIR/cases/$CASE" ] || { echo "找不到 cases/$CASE（workdir=$WORKDIR）" >&2; exit 2; }

# ---- 每 case 默认 size env（把"短核固定开销陷阱"处理在脚本内，agent 无需知晓）----
# 优先级：--size-env 命令行 > cases/<case>/bench.env 文件（case 自带，新 case 通用）> 内置默认。
# bench.env 内容形如一行： SIZE_ENV="RMS_B=32768"  —— agent 建短核 case 时按约定放此文件即可，脚本无需改。
if [ -z "$SIZE_ENV" ] && [ -f "$WORKDIR/cases/$CASE/bench.env" ]; then
  # shellcheck disable=SC1090
  SIZE_ENV="$(. "$WORKDIR/cases/$CASE/bench.env" 2>/dev/null; echo "${SIZE_ENV:-}")"
fi
if [ -z "$SIZE_ENV" ]; then
  case "$CASE" in
    rbf)        SIZE_ENV="RBF_SIZE=2048" ;;    # 非短核，与既有结果/config 默认一致
    layernorm)  SIZE_ENV="LN_B=32768" ;;       # 默认 B=4096 前向仅 0.06ms→加速比虚高；放大到 ≥0.2ms
    softmax_ce) SIZE_ENV="SMCE_B=8192" ;;      # 已 ≥0.2ms
    *)          SIZE_ENV="" ;;                 # 未知 case：靠 cases/<case>/bench.env 声明（见上）
  esac
fi
# 逗号转空格，供远程 env 使用（K=V,K=V → K=V K=V）
SIZE_ENV="${SIZE_ENV//,/ }"

# ---- (A) 防作弊前置：framework/ 不得被本地改动 ----
# 用 numstat -w 忽略纯行尾/空白差异（WSL↔Windows checkout 常见 CRLF 视图差异会误判为脏）；
# 有实际内容改动才拒。--quiet 不支持 -w，故用 numstat 是否有输出判断。
if git -C "$WORKDIR" rev-parse --git-dir >/dev/null 2>&1; then
  FW_DIFF="$(git -C "$WORKDIR" diff --numstat -w -- framework/ 2>/dev/null)"
  if [ -n "$FW_DIFF" ]; then
    echo "framework/ 有实质改动——评测基座只读，拒绝运行：" >&2
    echo "$FW_DIFF" >&2
    emit FRAMEWORK_DIRTY; exit 1
  fi
fi

# ---- (A2) 机械轮次上限兜底（Stage C 全自主防跑飞）----
# --round-cap N (N>0) 启用：每次调用对本 case 计数；超过 N 轮直接拒跑并发 ROUND_CAP_EXCEEDED，
# 即便 agent 失控也能在上下文里拿到硬停信号。计数存 workdir 的 .a100_round_<case>，PASS 时清零（见末尾）。
ROUND_FILE="$WORKDIR/.a100_round_$CASE"
if [ "$ROUND_CAP" -gt 0 ] 2>/dev/null; then
  ROUND_N=0; [ -f "$ROUND_FILE" ] && ROUND_N="$(cat "$ROUND_FILE" 2>/dev/null || echo 0)"
  ROUND_N=$((ROUND_N + 1))
  echo "$ROUND_N" > "$ROUND_FILE"
  if [ "$ROUND_N" -gt "$ROUND_CAP" ]; then
    echo "已达轮次上限 $ROUND_CAP（本轮第 $ROUND_N 次）——停止自主 loop，请人工介入。" >&2
    emit ROUND_CAP_EXCEEDED; exit 1
  fi
  echo "[run_on_a100] round $ROUND_N/$ROUND_CAP" >&2
fi

# ---- (B) 打包 payload（只含 cases/<case>；--sync-cli 时附带两个 CLI）----
PAYLOAD="$(mktemp -t nl2cuda_payload.XXXXXX.tgz)"
trap 'rm -f "$PAYLOAD"' EXIT
PACK_LIST=("cases/$CASE")
if [ "$SYNC_CLI" = "1" ]; then
  PACK_LIST+=("skill/scripts/verify_case.py" "skill/scripts/bench_case.py")
fi
tar czf "$PAYLOAD" -C "$WORKDIR" --exclude='__pycache__' --exclude='*.pyc' "${PACK_LIST[@]}" || { emit SSH_ERROR; exit 1; }

# ---- (可选) --auto-gpu：挑显存占用最低的卡 ----
if [ "$AUTO_GPU" = "1" ]; then
  PICK="$(ssh_a100 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | sort -t, -k2 -n | head -1 | cut -d, -f1' 2>/dev/null | tr -d ' \r')"
  [ -n "$PICK" ] && GPU="$PICK"
fi

echo "[run_on_a100] case=$CASE gpu=$GPU size_env='$SIZE_ENV' sync_cli=$SYNC_CLI workdir=$WORKDIR" >&2

# ---- (C)(D) 传输 + 远程执行，≤3 次重试 ----
REMOTE_PAYLOAD='~/.nl2cuda_payload.tgz'
RESULT=""
for attempt in 1 2 3; do
  # 先把 tar 经 stdin 送到远程文件（与下一步的 heredoc-stdin 分离，避免 stdin 争用）
  if ! cat "$PAYLOAD" | ssh_a100 "cat > $REMOTE_PAYLOAD" 2>/dev/null; then
    echo "[run_on_a100] 传输失败（第 $attempt 次），重试…" >&2; sleep 5; continue
  fi
  # 再跑远程评测：quoted heredoc（本地不展开），本地值经 bash -s 位置参数传入（无需转义 \$）
  RESULT="$(ssh_a100 "bash -s -- '$CASE' '$GPU' '$SIZE_ENV' '$REMOTE_REPO' '$REMOTE_PAYLOAD'" 2>/dev/null <<'REMOTE'
set -u
CASE="$1"; GPU="$2"; SIZE_ENV="$3"; REMOTE_REPO="$4"; PAYLOAD="$5"
REPO="$(eval echo "$REMOTE_REPO")"          # 展开 ~
PAYLOAD="$(eval echo "$PAYLOAD")"
export PATH="$HOME/miniconda3/bin:/usr/local/cuda/bin:$PATH"
cd "$REPO" || { echo "VERDICT=SSH_ERROR"; exit 0; }
# 预清旧 case 文件（删/改名的 .cu 不残留），再覆盖解包
rm -rf "cases/$CASE/kernels"/*.cu "cases/$CASE"/*.py 2>/dev/null || true
tar xzf "$PAYLOAD" || { echo "VERDICT=SSH_ERROR"; exit 0; }
# 清扩展缓存（含 case 名的都清，避免复用 stale .so；不误伤其他 case）
rm -rf "$HOME/.cache/torch_extensions"/*/*"$CASE"* 2>/dev/null || true
RUN="CUDA_VISIBLE_DEVICES=$GPU CUDA_ARCHS=80 $SIZE_ENV"
# verify 门禁（exit 0/1）→ 通过才 bench
if env $RUN python skill/scripts/verify_case.py --case "$CASE" > /tmp/nl2_verify.log 2>&1; then
  echo "---VERIFY---"; cat /tmp/nl2_verify.log
  env $RUN python skill/scripts/bench_case.py --case "$CASE" --emit-verdict > /tmp/nl2_bench.log 2>&1 || true
  echo "---BENCH---"; cat /tmp/nl2_bench.log
else
  echo "---VERIFY---"; cat /tmp/nl2_verify.log
  echo "VERDICT=VERIFY_FAIL"
fi
REMOTE
)"
  # 拿到含 VERDICT 的输出即成功
  if echo "$RESULT" | grep -q '^VERDICT='; then break; fi
  echo "[run_on_a100] 远程无 VERDICT 输出（第 $attempt 次），重试…" >&2; sleep 5
done

# ---- (E) 回传：原始日志（供诊断）+ 末行单条 VERDICT（机读）----
if [ -z "$RESULT" ] || ! echo "$RESULT" | grep -q '^VERDICT='; then
  echo "$RESULT"
  emit SSH_ERROR; exit 1
fi
echo "$RESULT"
FINAL_VERDICT="$(echo "$RESULT" | grep '^VERDICT=' | tail -1)"
echo "$FINAL_VERDICT"
# PASS 即 loop 完成，清零轮次计数（下次同 case 重新计）
if [ "$ROUND_CAP" -gt 0 ] 2>/dev/null && echo "$FINAL_VERDICT" | grep -q '^VERDICT=PASS'; then
  rm -f "$ROUND_FILE" 2>/dev/null || true
fi
# --strict：按 VERDICT 返回退出码（PASS=0，其余=1），供 aider --auto-test 等靠退出码驱动的自主闭环；
# 默认(非strict) 恒 exit 0（Stage B 人工读 VERDICT 决策，不希望非零退出打断脚本）。
if [ "$STRICT" = "1" ]; then
  echo "$FINAL_VERDICT" | grep -q '^VERDICT=PASS' && exit 0 || exit 1
fi
exit 0
