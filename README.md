# nl2cuda-kernel-agent

自然语言 → CUDA 前反向 Kernel 自动生成 **Skill**。

用户用自然语言描述一个算法结构（附张量 shape/dtype 约定），挂载了本 skill 的现成 agent（Claude Code / Codex / 开源 agent）自动完成闭环：

1. **PyTorch 参考实现** —— 由自然语言生成可运行的前反向参考，作为正确性金标准；
2. **CUDA Kernel 生成** —— 生成自定义前向 + 反向 kernel；
3. **优化迭代** —— 基于 profile 反馈做 tiling / float4 / warp 原语 / 算子融合等优化；
4. **交付** —— 产出可独立编译的 `.cu` 文件。

> 交付物是 **skill（方法论 + 工具）**，不是从零构建的 agent。详见 [工作目标.md](工作目标.md)（任务契约）与 [工作计划.md](工作计划.md)（执行清单）。

## 验收用例与结果

**成对距离 / RBF 高斯核矩阵**（非 attention），FP32，形状 **N=M=2048（默认；env `RBF_SIZE` 可切 4096）, D=64**：

- 前向：`K[i,j] = exp(-gamma · ||x_i − y_j||²)`，X:[N,D], Y:[M,D] → K:[N,M]
- 反向：对 X、Y 求梯度 dX、dY

**验收标准**：以 PyTorch 参考为金标准，≥5 组随机输入前反向均 `allclose(atol=rtol=1e-2)`；且自定义 CUDA kernel 相对 `torch.compile`（默认 mode）前向与反向均快 ≥5%。

**✅ 达标结果（A100-SXM4-40GB, sm_80）**：前向 **1.10×**、反向 **1.17×**，CV<1.5%，前反向 5 种子正确性全 PASS。全程 fp32、无 fast-math、无高层算子落回。可独立编译的纯 CUDA 交付版见 [`cases/rbf/delivery/`](cases/rbf/delivery/)（`make test` 一键独立编译+自测，不依赖 PyTorch）。

## 16 形态光谱（v1）

RBF 之后又扩了 15 个形态，用**纯自然语言输入**在三宿主（aider / gptme / codex）上逐一验证 skill 的通用性与边界。加速比列 codex 版（诚实 baseline 下的独立复验值，代表 skill 能达到的水平）：

| 形态 | 维度 | codex 前/反 | 光谱区间 |
|------|------|------|------|
| rbf | 距离/广播 | 1.10 / 1.17 | 稳赢 |
| layernorm | 归约 | 1.09 / 1.11 | 擦线 |
| softmax_ce | 归约 | 达标 | 擦线 |
| scan | 前缀扫描(CUB) | 3.4 / 4.1 | 稳赢 |
| gemm_bias_gelu | 矩阵乘+融合(cuBLAS) | 1.19 / 1.16 | 稳赢 |
| online_softmax | 单遍在线归约 | ~1.10 / 2.95 | 稳赢 |
| rope | 位置编码/elementwise | 2.54 / 1.18 | 稳赢 |
| linear_ssm | 固定系数递推→cumsum | 3.9 / 4.6 | 稳赢 |
| welford | 单遍统计→两遍归约 | 1.097 / 1.27 | 擦线 |
| causal_attn | attention 融合链 | 1.26~1.37 / 1.16~1.40 | 宿主分层(仅codex) |
| conv1d | 1D 因果卷积 | 4.78 / 1.86 | 稳赢 |
| gated_ssm | 变系数递推→O(T²)下三角 | 7.80 / 13.78 | 稳赢 |
| scatter_add | 数据依赖写+atomic | 2.19 / 1.14~1.22 | 宿主分层 |
| topk | 数据依赖控制流+稀疏反向 | 1.35 / 1.33 | 宿主分层 |
| maxpool | 2D 空间+argmax 稀疏反向 | 1.03(带宽墙) / 4.58 | 带宽墙(前向) |
| spmv | CSR 间接寻址/非规则访存 | 5.97 / 2.48 | 稳赢 |

**四光谱区间**：**稳赢区**（距离/scan/卷积/SSM/间接寻址——手写融合空间大）· **擦线区**（归一化/reduce 密集前向——torch.compile 已近最优，1.05~1.10 抖动）· **宿主分层区**（融合密集如 attention/scatter/topk——kernel 实现力或基线诚实度决定成败，常仅最强宿主 codex 拿下）· **带宽墙区**（纯访存低算术强度如 maxpool 前向——三宿主一致过不了，算法本征无优化空间，非 skill/agent 失败）。

**核心结论**：
- **三维独立**：skill 质量 / 模型能力 / 宿主架构自主性互不相同，测试须三维分离归因（同一 case codex 稳过、aider/gptme 可能因架构或能力栽）。
- **弱 baseline 5 变种**（reference 不诚实致加速比虚高）：① Python for 循环 ② O(N²) Toeplitz 伪向量化 ③ 规模专属慢分支 ④ cumprod/cumsum 数值脆弱+反向畸形 ⑤ 用更慢通用算子(sort)替代最优原语(topk)。均由 [`skill/scripts/check_reference.py`](skill/scripts/check_reference.py) 静态预检自动预警。
- **评测鲁棒性加固**：`run_on_a100.sh --auto-scale` 自适应放大到"计算主导区"（补短核假象）+ `check_reference.py` 静态扫危险写法（补弱 baseline）+ 防作弊红线 §1-5 + 擦线 3 连稳定判据。
- **红线准则**：torch 高层算子（`F.*`/`nn.*`/`matmul`/`sparse.mm`/SDPA）禁；CUDA 官方底层库（cuBLAS/CUB/cuSPARSE）允许；通用张量原语（`topk`/`sort`/`cumsum`/`scatter_add`）reference 里允许但 candidate 须手写 `.cu`。



两层架构：**通用框架**（算法无关）+ **算法 case**（每算法一份，可替换）。RBF 是第一个 case，后续加新算法只需新增 `cases/<name>/`。

| 路径 | 用途 |
|------|------|
| `framework/` | 通用框架（**算法无关**）：`protocol.py` 计时/容差协议、`case.py` Case 协议、`verify.py` 正确性验证、`bench.py` 计时、`loader.py` 编译加载、`smoke.cu` 冒烟。**评测脚本对 agent 只读、进程隔离** |
| `cases/<name>/` | 算法 case：`__init__.py`(暴露 CASE)、`reference.py` 金标准、`config.py` 形状/参数、`description.md` NL 描述、`kernels/` CUDA kernel、`op.py` autograd 封装；`delivery/` 可独立编译的纯 CUDA 交付版 |
| `skill/` | 交付的 skill：`DESIGN.md` 设计、`SKILL.md` 方法论、`loop.md` 迭代循环、`scripts/` 通用 CLI（`verify_case.py`/`bench_case.py`/`profile_case.py --case <name>`） |
| `notebooks/` | Colab 驱动 notebook（clone/pull + 装依赖 + 运行入口） |
| `scripts/` | 通用脚本（如 `probe_env.py` 环境探测） |

已验证的 case（**16 形态**，覆盖归约/矩阵乘/scan/卷积/位置编码/数据依赖/间接寻址等维度，framework 零改动即支持每一个）：`rbf` · `layernorm` · `softmax_ce` · `scan` · `gemm_bias_gelu` · `online_softmax` · `rope` · `linear_ssm` · `welford` · `causal_attn` · `conv1d` · `gated_ssm` · `scatter_add` · `topk` · `maxpool` · `spmv`。三宿主（aider/gptme/codex）逐形态对照与结论见 [`skill/AGENT_TEST_MATRIX.md`](skill/AGENT_TEST_MATRIX.md)。

## 挂到宿主 agent（通用性）

skill 是**宿主无关**的（方法论用自然语言写、工具是独立 python）。挂载方式：
- **Claude Code**（已验证）：把 `skill/SKILL.md` 作为 skill 载入；agent 读它执行 NL→torch→CUDA→verify→bench→优化循环。
- **Codex / 开源 agent**：同样把 `skill/SKILL.md` 作为系统提示/技能文档提供；工具 `skill/scripts/*.py` 是纯 python CLI，任何能执行 shell 的 agent 都能调用。
- **新增算法**：复制 `cases/rbf/` 为 `cases/<name>/`，替换 `reference.py`/`config.py`/`description.md`/`__init__.py` 的 CASE（`grad_inputs`/`params`），清空 `kernels/` 重写；framework 与 CLI 无需改动，`--case <name>` 即复用整套验证/计时。

## 开发 / 验收环境

- 开发机无 NVIDIA GPU，仅写代码 + git。
- GPU 编译/运行/验收：初期用 **Google Colab**（T4, sm_75）跑通全流程；达标验收在 **NVIDIA A100-SXM4-40GB（sm_80）** 上完成。
- `framework/loader.py` 支持多架构编译（env `CUDA_ARCHS`，默认 `75,80`）。
- 本仓库为 public，`git clone` 无需认证。

## 快速开始（在 Colab）

```bash
!git clone https://github.com/SilenceWanna/nl2cuda-kernel-agent.git
%cd nl2cuda-kernel-agent
!python scripts/probe_env.py   # 确认 GPU / CUDA / PyTorch
```

后续验证与基准入口见 `notebooks/run.ipynb`。
