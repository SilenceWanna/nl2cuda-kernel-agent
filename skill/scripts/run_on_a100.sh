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
# VERDICT 文法：PASS · PASS_SCALE_SUSPECT（擦线达标但放大规模掉破1.05=规模挑选嫌疑，strict下算未达标）· VERIFY_FAIL · BENCH_FAIL · CV_INVALID · FRAMEWORK_DIRTY · SSH_ERROR
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
AUTO_SCALE=1                                 # 1=默认开：无 size-env/bench.env 时自动探短核并放大重测(harness 兜底，补 agent 不建 bench.env 的弱点)
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
    --no-auto-scale) AUTO_SCALE=0; shift ;;
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

# ---- 自动放大兜底：无 size-env/bench.env 时，grep config 发现规模 env 变量，交远程探短核后自适应放大 ----
# 补 aider 弱点：它不主动建 bench.env → 短核 case 会被固定开销虚高骗。harness 侧兜底，宿主无关。
# 只在"没有任何显式规模指定"时启用；显式 --size-env / bench.env / 内置表命中则不介入。
# 自适应放大（2026-07-17，替代旧固定 32768）：探到短核后迭代放大规模（×4/轮），直到 baseline 前反向
# 均进入"计算主导区"（≥AUTO_TARGET_MS）或触规模/轮次上限——避免固定值对不同算法复杂度不适配
# （旧固定 32768 曾坑：attention O(T²) 过大 283ms、scatter O(N) 仍短核 CV_INVALID、conv1d 阈值边界）。
SCALE_VAR=""
AUTO_START=8192      # 放大起点规模
AUTO_MAX=4194304     # 规模上限（防显存爆炸/失控，4M）
AUTO_TARGET_MS=1.0   # 目标：baseline 前反向均 ≥此耗时(ms) 才算进入计算主导区（CV 稳、摊薄固定开销）；未达则放大
if [ "$AUTO_SCALE" = "1" ] && [ -z "$SIZE_ENV" ]; then
  # 取 config.py 第一个 os.environ.get("XXX", ...) 的变量名（如 RMS_B / SCAN_B / LN_B）
  SCALE_VAR="$(grep -oE 'os\.environ\.get\("[A-Za-z_]+"' "$WORKDIR/cases/$CASE/config.py" 2>/dev/null \
               | head -1 | sed -E 's/.*"([A-Za-z_]+)".*/\1/')"
  # 数一共有几个规模 env 变量：≥3 个 = 多维 config（N×C×H×W…），auto-scale 只放大第一个会病态。
  SCALE_VAR_COUNT="$(grep -oE 'os\.environ\.get\("[A-Za-z_]+"' "$WORKDIR/cases/$CASE/config.py" 2>/dev/null | wc -l | tr -d ' ')"
  if [ -z "$SCALE_VAR" ]; then
    echo "[run_on_a100] 提示：无 size-env/bench.env 且 config 未参数化规模，无法自动放大——结果可能是短核虚高。" >&2
  elif [ "${SCALE_VAR_COUNT:-1}" -ge 3 ] 2>/dev/null; then
    # 多维乘积规模 case（如 grid_sample N×C×H×W×OH×OW、GroupNorm N×C×H×W）：auto-scale 只放大首变量
    # $SCALE_VAR 会造成病态形状（batch 巨大、空间小 → 前向假暴/加速比失真）或爆显存（多维乘积 OOM）。
    # 强烈建议 agent 建 bench.env 声明**平衡**的计算主导区规模。此处仍会兜底放大首变量，但警示风险。
    echo "[run_on_a100] ⚠️ 多维规模 config（探到 $SCALE_VAR_COUNT 个规模变量,首=$SCALE_VAR）：auto-scale 只放大首变量易病态形状(batch大空间小→加速比失真)或OOM。**强烈建议建 cases/$CASE/bench.env 声明平衡规模**(如各维适度放大而非单放 batch)。当前无 bench.env→仍兜底放大 $SCALE_VAR,结果可能失真,请以平衡 bench.env 规模为准。" >&2
  else
    echo "[run_on_a100] auto-scale 待命：若探到短核，将从 $SCALE_VAR=$AUTO_START 起自适应放大(×4/轮)到 baseline≥${AUTO_TARGET_MS}ms 或上限 $AUTO_MAX。" >&2
  fi
fi

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

# ---- (A3) reference 静态预检（本地，跑 A100 前）：扫 reference/make_inputs 弱 baseline 危险写法 ----
# 弱 baseline 源头在 reference 写法(for/T²-Toeplitz/规模分支/cumprod 脆弱/挑输入分布/高层算子)。
# auto-scale 管"规模不够"、本预检管"写法可疑"，正交互补，在传输/编译前就预警(省算力)。
# 默认只 WARN 打印不拦(有合法例外如变系数递推 T²)；红线(高层算子)命中才在日志显著标红，仍交 verify 定论。
if [ -f "$SCRIPT_DIR/check_reference.py" ]; then
  REF_SCAN="$(python "$SCRIPT_DIR/check_reference.py" --case "$CASE" --workdir "$WORKDIR" 2>/dev/null)"
  if ! echo "$REF_SCAN" | grep -q 'REF_CHECK=CLEAN'; then
    echo "[run_on_a100] reference 预检发现可疑写法（不拦，供核查——弱 baseline 常源于此）：" >&2
    echo "$REF_SCAN" | grep -E '^\s*\[(WARN|RED)\]' >&2
  fi
fi

# ---- (A2) 轮次计数 + 机械上限兜底 ----
# **总是计数**每次自测调用（存 workdir 的 .a100_round_<case>），用于客观统计"到达标跑了几轮"——
# 不依赖 --round-cap（后者只额外管"超 N 轮拒跑"）。PASS 时把最终轮次存档到 .round_final_<case> 再清零（见末尾）。
ROUND_FILE="$WORKDIR/.a100_round_$CASE"
ROUND_N=0; [ -f "$ROUND_FILE" ] && ROUND_N="$(cat "$ROUND_FILE" 2>/dev/null || echo 0)"
ROUND_N=$((ROUND_N + 1))
echo "$ROUND_N" > "$ROUND_FILE"
if [ "$ROUND_CAP" -gt 0 ] 2>/dev/null; then
  if [ "$ROUND_N" -gt "$ROUND_CAP" ]; then
    echo "已达轮次上限 $ROUND_CAP（本轮第 $ROUND_N 次）——停止自主 loop，请人工介入。" >&2
    emit ROUND_CAP_EXCEEDED; exit 1
  fi
  echo "[run_on_a100] round $ROUND_N/$ROUND_CAP" >&2
else
  echo "[run_on_a100] round $ROUND_N（本 case 累计自测次数）" >&2
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
  RESULT="$(ssh_a100 "bash -s -- '$CASE' '$GPU' '$SIZE_ENV' '$REMOTE_REPO' '$REMOTE_PAYLOAD' '$SCALE_VAR' '$AUTO_START' '$AUTO_MAX' '$AUTO_TARGET_MS'" 2>/dev/null <<'REMOTE'
set -u
CASE="$1"; GPU="$2"; SIZE_ENV="$3"; REMOTE_REPO="$4"; PAYLOAD="$5"; SCALE_VAR="$6"; AUTO_START="$7"; AUTO_MAX="$8"; AUTO_TARGET_MS="$9"
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
  # 自适应放大兜底：无显式 size-env 且发现规模变量时，只要 baseline 未进入"计算主导区"（前/反向任一
  # <AUTO_TARGET_MS）就迭代放大规模（从 AUTO_START 起 ×4/轮），直到前反向均 ≥AUTO_TARGET_MS 或触规模上限。
  # 用统一的"计算主导区"判据（而非旧 0.15ms 中间阈值——会踩 conv1d/scatter 险过边界的坑）。verify 不受规模影响。
  parse_ms() { grep -E "^\s*$1\s*:" /tmp/nl2_bench.log | head -1 | grep -oE '[0-9.]+ ms' | grep -oE '[0-9.]+'; }
  if [ -z "$SIZE_ENV" ] && [ -n "$SCALE_VAR" ]; then
    BF="$(parse_ms forward)"; BB="$(parse_ms backward)"
    if awk "BEGIN{exit !(($BF+0) < $AUTO_TARGET_MS || ($BB+0) < $AUTO_TARGET_MS)}" 2>/dev/null; then
      echo "[auto-scale] baseline fwd=${BF}ms/bwd=${BB}ms 未达计算主导区(≥${AUTO_TARGET_MS}ms) → 自适应放大"
      SCALE="$AUTO_START"; ITER=0
      # PREV_OK_LOG：上一个"成功产出有效计时"的规模日志（供 OOM/崩溃回退用）。
      # 多维乘积规模 case（如 GroupNorm N×C×H×W）放大单一变量会使总张量爆炸 → CUDA OOM，
      # 此时本轮 bench 拿不到有效 ms（BF/BB 空）——回退到上一成功规模，避免整轮 SSH_ERROR。
      PREV_OK_LOG=""; PREV_OK_SCALE=""; HALVE_TRIES=0
      while [ "$ITER" -lt 6 ]; do
        ITER=$((ITER+1))
        RUN2="CUDA_VISIBLE_DEVICES=$GPU CUDA_ARCHS=80 $SCALE_VAR=$SCALE"
        rm -rf "$HOME/.cache/torch_extensions"/*/*"$CASE"* 2>/dev/null || true
        env $RUN2 python skill/scripts/bench_case.py --case "$CASE" --emit-verdict > /tmp/nl2_bench.log 2>&1 || true
        BF="$(parse_ms forward)"; BB="$(parse_ms backward)"
        # OOM/崩溃探测：放大后拿不到有效计时（BF 或 BB 空）→ 多半是显存爆炸（多维乘积规模）。
        if [ -z "$BF" ] || [ -z "$BB" ]; then
          if grep -qiE "out of memory|CUDA error|RuntimeError" /tmp/nl2_bench.log 2>/dev/null; then
            echo "[auto-scale] 第${ITER}轮 $SCALE_VAR=$SCALE bench 无有效计时（疑似 OOM/崩溃，多维规模放大过大）"
          else
            echo "[auto-scale] 第${ITER}轮 $SCALE_VAR=$SCALE bench 无有效计时"
          fi
          if [ -n "$PREV_OK_LOG" ]; then
            # 有上一成功规模 → 回退用它的结果（多维 case 放大到 OOM 前的最大可行规模）
            cp "$PREV_OK_LOG" /tmp/nl2_bench.log
            echo "[auto-scale] 回退到上一成功规模 $SCALE_VAR=$PREV_OK_SCALE 的结果（避免 OOM 致整轮失败）"
            break
          fi
          # 首轮起点就 OOM（AUTO_START 对该多维 case 过大）→ 减半重试找可行起点，不占放大轮次
          if [ "$SCALE" -gt 64 ] && [ "$HALVE_TRIES" -lt 8 ]; then
            HALVE_TRIES=$((HALVE_TRIES+1)); ITER=$((ITER-1)); SCALE=$((SCALE/2))
            echo "[auto-scale] 起点规模过大致 OOM，减半重试 → $SCALE_VAR=$SCALE"
            continue
          fi
          echo "[auto-scale] 无可行规模（减半到 $SCALE_VAR=$SCALE 仍 OOM），放弃放大用原始结果"
          break
        fi
        echo "[auto-scale] 第${ITER}轮 $SCALE_VAR=$SCALE → baseline fwd=${BF}ms/bwd=${BB}ms"
        # 记为上一成功规模（供下一轮 OOM 回退）
        PREV_OK_LOG="/tmp/nl2_bench_ok_${SCALE}.log"; cp /tmp/nl2_bench.log "$PREV_OK_LOG"; PREV_OK_SCALE="$SCALE"
        # 计算主导区：前反向均 ≥目标耗时 → 停（结果可信）
        if awk "BEGIN{exit !(($BF+0) >= $AUTO_TARGET_MS && ($BB+0) >= $AUTO_TARGET_MS)}" 2>/dev/null; then
          echo "[auto-scale] 已进入计算主导区(前反向均≥${AUTO_TARGET_MS}ms)，用此规模结果"; break
        fi
        # 触规模上限 → 停（已尽力放大，用当前结果）
        NEXT=$((SCALE*4))
        if [ "$NEXT" -gt "$AUTO_MAX" ]; then
          echo "[auto-scale] 触规模上限 $AUTO_MAX，停止放大，用当前规模结果"; break
        fi
        SCALE="$NEXT"
      done
    fi
  elif [ -n "$SIZE_ENV" ]; then
    # 有显式 size-env/bench.env 时不放大（尊重 agent 声明的规模），但要防"规模挑选"作弊——
    # agent 可能（有意或无意）挑一个 baseline 偏短的规模让固定开销撑高加速比（短核假象的宿主自选变种）。
    # —— 实测暴露（GroupNorm gptme）：挑 GN_N=384（baseline 前向仅 0.92ms<1ms）前向擦线 1.07 PASS，
    #    放大到计算主导区（768/1536）前向掉到 1.048/1.035 FAIL——是规模挑选，非真实优势。
    # 主动检测：擦线 PASS + 短核时，自动换 2×/4× 规模复测；若擦线侧在放大规模下掉破 1.05 →
    # 标记 SCALE_SUSPECT（把我此前手动多规模复验的拆穿能力内化进 harness）。codex 那种跨规模稳赢不触发。
    # 加速比从 VERDICT 行解析（bench.log 里 "forward :" 出现 3 次：baseline/candidate/加速比，
    # 易取错行；VERDICT=PASS fwd=1.07x bwd=1.06x 唯一且明确）。
    BF="$(parse_ms forward)"; BB="$(parse_ms backward)"
    VLINE="$(grep '^VERDICT=' /tmp/nl2_bench.log | head -1)"
    CUR_VERDICT="$(echo "$VLINE" | grep -oE 'VERDICT=[A-Z_]+' | cut -d= -f2)"
    SF="$(echo "$VLINE" | grep -oE 'fwd=[0-9.]+' | cut -d= -f2)"
    SB="$(echo "$VLINE" | grep -oE 'bwd=[0-9.]+' | cut -d= -f2)"
    # 触发条件：当前 PASS + 有短核侧(<AUTO_TARGET_MS) + 该短核侧擦线(<1.15×)。三者皆满足才复测。
    SHORT_BRITTLE=0
    if [ "$CUR_VERDICT" = "PASS" ] && [ -n "$BF" ] && [ -n "$BB" ] && [ -n "$SF" ] && [ -n "$SB" ]; then
      awk "BEGIN{f=($BF<$AUTO_TARGET_MS && $SF<1.15); b=($BB<$AUTO_TARGET_MS && $SB<1.15); exit !(f||b)}" 2>/dev/null && SHORT_BRITTLE=1
    fi
    if [ "$SHORT_BRITTLE" = "1" ]; then
      # 解析 SIZE_ENV 第一个变量做 ×2/×4 放大复测（同 auto-scale 抓变量法）
      PICK_VAR="$(echo "$SIZE_ENV" | grep -oE '[A-Za-z_]+=' | head -1 | tr -d =)"
      PICK_VAL="$(echo "$SIZE_ENV" | grep -oE "${PICK_VAR}=[0-9]+" | head -1 | cut -d= -f2)"
      echo "[规模敏感复测] 擦线 PASS(fwd=${SF}x/bwd=${SB}x) 且指定规模 '$SIZE_ENV' baseline 偏短(fwd=${BF}ms/bwd=${BB}ms<${AUTO_TARGET_MS}ms) → 自动放大复测 $PICK_VAR"
      SUSPECT=0
      if [ -n "$PICK_VAR" ] && [ -n "$PICK_VAL" ]; then
        for MUL in 2 4; do
          BIG=$((PICK_VAL * MUL))
          BIG_ENV="$(echo "$SIZE_ENV" | sed -E "s/${PICK_VAR}=[0-9]+/${PICK_VAR}=${BIG}/")"
          rm -rf "$HOME/.cache/torch_extensions"/*/*"$CASE"* 2>/dev/null || true
          env CUDA_VISIBLE_DEVICES=$GPU CUDA_ARCHS=80 $BIG_ENV python skill/scripts/bench_case.py --case "$CASE" --emit-verdict > /tmp/nl2_big.log 2>&1 || true
          BIG_VLINE="$(grep '^VERDICT=' /tmp/nl2_big.log | head -1)"
          BSF="$(echo "$BIG_VLINE" | grep -oE 'fwd=[0-9.]+' | cut -d= -f2)"
          BSB="$(echo "$BIG_VLINE" | grep -oE 'bwd=[0-9.]+' | cut -d= -f2)"
          BV="$(echo "$BIG_VLINE" | grep -oE 'VERDICT=[A-Z_]+' | cut -d= -f2)"
          if [ -z "$BSF" ] || [ -z "$BSB" ]; then
            echo "[规模敏感复测] ${PICK_VAR}=${BIG}(×${MUL}) 无有效计时(疑似OOM)，跳过该规模"
            continue
          fi
          echo "[规模敏感复测] ${PICK_VAR}=${BIG}(×${MUL}) → fwd=${BSF}x/bwd=${BSB}x verdict=$BV"
          # 放大后任一侧掉破 1.05 → 前向达标存疑（原擦线 PASS 靠短核固定开销虚高，与是否"主动挑规模"无关）
          awk "BEGIN{exit !(($BSF+0)<1.05 || ($BSB+0)<1.05)}" 2>/dev/null && SUSPECT=1
        done
      fi
      if [ "$SUSPECT" = "1" ]; then
        echo "[规模敏感复测] ⚠️ 判定 SCALE_SUSPECT：原规模擦线 PASS 但放大到计算主导区后掉破 1.05——加速比强规模依赖=短核固定开销虚高，非真实 kernel 优势（存疑，非作弊指控：固定 bench.env 的历史短核 case 与主动挑规模都会命中，都应在计算主导区复核）。"
        EMIT_SUSPECT=1   # 延迟到 bench 日志 cat 之后再 emit（否则被日志里的原始 VERDICT=PASS 覆盖，tail -1 取错）
      else
        echo "[规模敏感复测] 放大规模后加速比仍稳(≥1.05)，非短核虚高——诚实达标。"
      fi
    fi
  fi
  echo "---BENCH---"; cat /tmp/nl2_bench.log
  # 规模挑选嫌疑：最后 emit（覆盖日志里的原始 VERDICT=PASS，使外层 tail -1 取到本行）
  if [ "${EMIT_SUSPECT:-0}" = "1" ]; then
    echo "VERDICT=PASS_SCALE_SUSPECT fwd=${SF}x bwd=${SB}x cv_ok=1"
  fi
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
# PASS 即 loop 完成：把"到达标的总轮次"存档到 .round_final_<case>（客观计数，供统计各宿主用了几轮），再清零计数
# 精确匹配"干净 PASS"（PASS 后跟空格或行尾）——排除 PASS_SCALE_SUSPECT（规模挑选嫌疑，不算达标、不存档轮次）。
if echo "$FINAL_VERDICT" | grep -qE '^VERDICT=PASS( |$)'; then
  echo "$ROUND_N" > "$WORKDIR/.round_final_$CASE" 2>/dev/null || true
  echo "[run_on_a100] 达标！到达标累计自测 $ROUND_N 轮（已存档 .round_final_$CASE）" >&2
  rm -f "$ROUND_FILE" 2>/dev/null || true
elif echo "$FINAL_VERDICT" | grep -q '^VERDICT=PASS_SCALE_SUSPECT'; then
  echo "[run_on_a100] ⚠️ PASS_SCALE_SUSPECT：擦线达标存疑（放大规模掉破 1.05=短核固定开销虚高）——不算达标，请在计算主导区规模优化 kernel 本体到稳过。" >&2
fi
# --strict：按 VERDICT 返回退出码（干净 PASS=0，其余含 SCALE_SUSPECT=1）——逼 agent 继续优化真实 kernel，
# 而非靠挑短核规模凑擦线蒙混过关（自主闭环下 SCALE_SUSPECT 会驱动它继续改，而挑更大规模也过不了）。
# 默认(非strict) 恒 exit 0（Stage B 人工读 VERDICT 决策，不希望非零退出打断脚本）。
if [ "$STRICT" = "1" ]; then
  echo "$FINAL_VERDICT" | grep -qE '^VERDICT=PASS( |$)' && exit 0 || exit 1
fi
exit 0
