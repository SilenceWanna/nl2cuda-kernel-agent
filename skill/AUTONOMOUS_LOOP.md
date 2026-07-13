# 自主闭环：Agent 直连 GPU 自测自优化（阶段 6）

本文档给**驱动 agent**（gptme / aider / …）一份可直接照做的自主闭环规范：
agent 自己调 `run_on_a100.sh` 在远程 A100 上跑 verify/bench，读机读 `VERDICT`，
按 [loop.md](loop.md) 迭代优化 kernel，直到达标或触及轮次上限。**无需人工搬运代码或跑评测。**

配合工具：`skill/scripts/run_on_a100.sh`（搬运+远程自测）、`skill/scripts/bench_case.py --emit-verdict`（机读判定）、
`skill/loop.md`（优化循环规范）、`framework/`（只读评测基座）。

## 前置（一次性，按驱动环境）

**驱动 agent 所在环境必须能双跳 SSH 到 A100**（密钥 `~/.ssh/nl2cuda_gpu`）：
- **aider（Windows Git Bash）**：密钥已在 Windows `~/.ssh/`，直接可用。
- **gptme（WSL）**：WSL 默认没有该密钥，需先拷入并锁权限：
  ```bash
  cp /mnt/c/Users/<你的用户名>/.ssh/nl2cuda_gpu ~/.ssh/ && chmod 600 ~/.ssh/nl2cuda_gpu
  ```
  （密钥仅本地使用，切勿提交进仓库。）
- **codex（Windows 或 WSL 均可，见下 §codex 接入）**：Windows 则同 aider（免拷）；WSL 则同 gptme（先拷密钥）。

**首次运行带 `--sync-cli`**：把带 `--emit-verdict` 的 `bench_case.py` 同步到 A100 一次。之后每轮省略。

## VERDICT 文法（agent 的唯一决策信号）

`run_on_a100.sh` 末行必打印一条 `VERDICT=...`，agent 只依据它决策：

| VERDICT | 含义 | agent 该做什么 |
|---------|------|----------------|
| `PASS` | 正确性 + 前反向加速比均达标 | **停止**，进入交付；本 case 完成 |
| `VERIFY_FAIL` | 正确性未过（allclose 失败） | **修正确性 bug**（读 `---VERIFY---` 日志的 err）。**不要看 bench 数**，正确性优先 |
| `BENCH_FAIL` | 正确但前向或反向 <1.05× | 按 loop.md 优化未达标那一侧的 kernel（先 profile 再对症） |
| `CV_INVALID` | 测量 CV>5%（共享卡噪声） | **原样重跑**（别改代码、别"优化"噪声）；连续多次则换空卡/等时段 |
| `FRAMEWORK_DIRTY` | 本地改了 framework/ | 撤销对 framework/ 的改动（评测基座只读，防作弊红线） |
| `SSH_ERROR` | 双跳/传输失败 | 重试；持续失败报告人工 |

`VERDICT` 行还带 `fwd=<x> bwd=<x> cv_ok=<0/1>`，供 agent 判断优化哪一侧、进展如何。

## 闭环循环体（每轮）

```
1. 调工具   → bash skill/scripts/run_on_a100.sh <case> --gpu 7
              （首轮加 --sync-cli）
2. 读 VERDICT → 按上表决策
3. 若 PASS   → 停止，交付
   若 *_FAIL → 读 ---VERIFY---/---BENCH--- 原始日志诊断，
               只改 cases/<case>/kernels/*.cu（及 op.py 封装），
               不动 framework/、不降精度、不改评测脚本。
4. 回到 1（重跑同一命令验证本轮改动）
```

## 终止条件（满足任一即停，遵 loop.md）

- ✅ `VERDICT=PASS` → 交付。
- ⏹ **轮次上限 8–12 轮**仍未达标 → 停下，报告当前最好结果 + 瓶颈分析，请人工介入。
- ⏹ **连续 2–3 轮无有效提升**（排除 CV 噪声后）→ 停下报告。

> 半自动阶段（Stage B）：人工盯每轮，可随时喊停。
> 全自主阶段（Stage C）：给 `run_on_a100.sh` 传 `--round-cap N`（如 12）启用**机械轮次兜底**——脚本按 case
> 计数（存 workdir 的 `.a100_round_<case>`，已 gitignore），超过 N 轮直接拒跑并发 `VERDICT=ROUND_CAP_EXCEEDED`，
> 即便 agent 失控也能在上下文拿到硬停信号；`VERDICT=PASS` 时自动清零计数。默认 `--round-cap 0`（禁用，即 Stage B 行为）。

## Stage C 全自主启动（去人工中转）

前提同上（能双跳 SSH + 有脚本 + 首轮 --sync-cli）。与 Stage B 的唯一区别：kickoff 提示里让 agent
**每轮命令都带 `--round-cap 12`**，并说明 `ROUND_CAP_EXCEEDED` = 硬停。agent 自主循环到 PASS 或触顶，无需人工每轮确认。

```
你将全自主优化 cases/<case>/ 到达标，无人工介入。严格遵守 skill/AUTONOMOUS_LOOP.md 与 loop.md。
每轮：跑 `bash skill/scripts/run_on_a100.sh <case> --gpu 7 --round-cap 12`（第一轮加 --sync-cli），读末行 VERDICT：
PASS→停并交付；VERIFY_FAIL→读日志修正确性(不看bench)；BENCH_FAIL→按loop.md优化未达标侧kernel；
CV_INVALID→原样重跑；ROUND_CAP_EXCEEDED→已达上限，立即停止并报告当前最好结果+瓶颈,不要再跑。
只改 cases/<case>/kernels/*.cu 及 op.py，绝不动 framework/、不降精度、不改评测脚本。现在开始。
```

> 建议 Stage C 首个 case 仍在旁观察（不干预、只看它是否自主收敛/正确触顶），确认全自主行为无误后再放手批量。

## 纪律（防作弊红线，不可违反）

- **正确性优先**：`VERIFY_FAIL` 时先修正确性，绝不看 bench 数、绝不为凑速度降精度。
- **framework/ 只读**：只改 `cases/<case>/`；改了 framework/ 会被 `run_on_a100.sh` 前置拒（`FRAMEWORK_DIRTY`）。
- **不信噪声**：`CV_INVALID` 原样重测，不基于噪声数据做优化决策。
- **fp32 全精度**：不用 fast-math、不降精度换速度（降精度必挂 verify）。

## kickoff 提示（复制给 agent，替换 `<case>`）

```
你将用自主闭环优化 cases/<case>/ 的 CUDA kernel 到达标。严格遵守 skill/AUTONOMOUS_LOOP.md 与 skill/loop.md。
每轮：调 `bash skill/scripts/run_on_a100.sh <case> --gpu 7`（第一轮加 --sync-cli），
读末行 VERDICT 决策——PASS 则停并交付；VERIFY_FAIL 则读 ---VERIFY--- 日志修正确性 bug（不看 bench）；
BENCH_FAIL 则按 loop.md 优化未达标那侧的 kernel（先诊断瓶颈）；CV_INVALID 原样重跑。
只改 cases/<case>/kernels/*.cu 及 op.py，绝不动 framework/、不降精度、不改评测脚本。
8-12 轮仍不达标就停下报告瓶颈。现在开始第一轮。
```

## codex 接入（阶段6 方案；样本：codex 自己的 LayerNorm 前向 0.97×）

codex 与 aider/gptme 的闭环规范完全一致（同一份 `run_on_a100.sh` + VERDICT + loop.md），
仅在**代理配置**与**执行方式**上有 codex 特有点，下面写清。样本选 codex 自己重生的 LayerNorm——
它正确性全 PASS、反向达标，**只差前向**（放大规模后 0.97×，短核 tiling 不足），是**第三种失败路径
（BENCH_FAIL 之前向优化）**，与 gptme(反向性能)、aider(正确性纠错) 互补。

### 代理配置（codex 特有，重要）
- codex 实际使用走**用户自配的外部代理**，非 aider/gptme 用的 dongcc(`127.0.0.1:8787`)：
  `~/.codex/config.toml` 的 `[model_providers.OpenAI]`（`base_url="https://codex.0u0o.com/v1"`,
  `wire_api="responses"`）。用时把顶部 `model_provider` 切到 `"OpenAI"`（当前为测 aider/gptme 临时设的 `"dongcc"`）。
- **切代理与阶段6工具链无关**——`run_on_a100.sh` 是纯 shell + SSH，不经模型代理；换 provider 只影响 codex 自身的 LLM 调用。

### 环境与前置（Win / WSL 两种都可）
- **codex 在 Windows**：密钥 `~/.ssh/nl2cuda_gpu` 已在，`run_on_a100.sh` 直接可用（同 aider，免拷密钥）。
- **codex 在 WSL**：先 `cp /mnt/c/Users/<user>/.ssh/nl2cuda_gpu ~/.ssh/ && chmod 600`（同 gptme，已验证 WSL 能双跳）。
- 工作目录：checkout codex 的 LayerNorm 分支（`test/kt-layernorm-codex`）到 codex workdir；确保有 `run_on_a100.sh`
  与带 `--emit-verdict` 的 `bench_case.py`（从主仓库拷入或 pull main）。首轮带 `--sync-cli`。
- codex 的 config.py 已支持 `LN_B` env（`run_on_a100.sh` 默认 `LN_B=32768`，避短核 CV 噪声）。

### 执行方式（codex 特有）
- codex 有内置 shell / `apply_patch` 工具与自己的审批模型：让它自主跑 `bash skill/scripts/run_on_a100.sh layernorm --gpu 7`
  （首轮 `--sync-cli`），stdout 的 `---BENCH---` 日志 + 末行 `VERDICT=` 进 codex 上下文驱动下一轮。
- 若 codex 环境限制不能直接执行 shell（类似 aider 无 /run 的情况）：退回**半自动中转**——codex 改 `cases/layernorm/kernels/layernorm_forward.cu`，人工替它跑 `run_on_a100.sh` 并把 VERDICT+日志回喂。
- 预期起点 `VERDICT=BENCH_FAIL fwd≈0.89x bwd≈1.22x`（反向已达标，前向短板）→ codex 按 loop.md 优化前向
  （加 shared-mem tiling / float4 / 提 occupancy，参考它自己 RBF 前向那套）→ 冲 `VERDICT=PASS`。

### kickoff 提示（codex，前向优化路径）
```
你将用自主闭环把 cases/layernorm/ 优化到达标。先读 skill/AUTONOMOUS_LOOP.md 和 skill/loop.md。
每轮：跑 `bash skill/scripts/run_on_a100.sh layernorm --gpu 7`（第一轮加 --sync-cli），读末行 VERDICT。
当前正确性已 PASS、反向达标，预期 VERDICT=BENCH_FAIL 且 fwd<1.05（前向短板）。
BENCH_FAIL 时按 loop.md 优化前向 kernel（cases/layernorm/kernels/layernorm_forward.cu）：
当前前向是朴素 block-per-row+warp规约，缺 tiling/向量化；参考你自己 RBF 前向的 shared-mem tiling 手法提速。
只改 cases/layernorm/kernels/*.cu 及 op.py，绝不动 framework/、不降精度、不改评测脚本。
CV_INVALID 则原样重跑。8-12 轮内冲到 VERDICT=PASS。现在开始第一轮。
```

> 注意：本节为**方案设计**（用户要求先设计、暂不用当前 dongcc 配置测试）。实跑前需先把 codex 的 `model_provider` 切回 `"OpenAI"`（用户自配代理），再按上述启动。
