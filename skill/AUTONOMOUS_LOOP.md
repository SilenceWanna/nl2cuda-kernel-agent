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

> 半自动阶段：人工盯每轮，可随时喊停。全自动阶段：脚本会在超上限时发 `VERDICT=ROUND_CAP_EXCEEDED` 兜底。

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
