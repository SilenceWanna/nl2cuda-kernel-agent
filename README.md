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

## 23 形态光谱（v1）

RBF 之后又扩了 22 个形态，用**纯自然语言输入**在三宿主（aider / gptme / codex）上逐一验证 skill 的通用性与边界。加速比列 codex 版（诚实 baseline 下的独立复验值，代表 skill 能达到的水平）。下表 25 行 = 23 个形态 + 2 个加固衍生 case（`temperature_softmax`/`l2norm_scale`，softmax/归一化家族内为验证确认闸门与反向融合手段而生，不单列形态序号）：

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
| cosine_sim | 成对余弦/点积+归一化 | 2.12 / 1.42 | 稳赢 |
| temperature_softmax | 带温度 softmax/归约 | 1.11 / 1.70 | 擦线(宿主分层) |
| l2norm_scale | L2 归一化+缩放/归约 | 1.01(带宽墙) / 1.48 | 带宽墙(前向) |
| groupnorm | 分组归约 | 1.28 / 1.32 | 稳赢(大 reduce) |
| geglu | 融合逐元素算子链 | 1.02(带宽墙) / 2.36 | 带宽墙(前向)/反向真赢 |
| gridsample | 数据依赖空间采样+atomic 散回 | 2.11 / 2.26 | 稳赢 |
| segment_softmax | 数据依赖变长分段(GNN) | 1.34 / 2.12 | 稳赢 |
| tridiag | 串行依赖线性求解(Thomas) | 1.27 / 9.80 | 稳赢(反向=解伴随) |
| cholesky | 稠密分解(挑战 cuSOLVER) | 0.31(厂商库墙) / 1.25 | 厂商库墙(前向)/反向真赢 |

**五光谱区间**：**稳赢区**（距离/scan/卷积/SSM/间接寻址/点积/大 reduce/数据依赖采样/变长分段——手写融合空间大 或 用了 autograd 不知道的解析结构）· **擦线区**（小 reduce 归一化前向如 LayerNorm/welford/temperature_softmax——torch.compile 已近最优，1.05~1.10 抖动、多规模易掉，须计算主导区 3 连复验）· **宿主分层区**（融合密集如 attention/scatter/topk/GEMM——kernel 实现力或基线诚实度决定成败，常仅最强宿主 codex 拿下）· **带宽墙区**（纯访存低算术强度前向如 maxpool/geglu/l2norm——candidate 访存量=baseline、都达峰值带宽，三宿主一致过不了，算法本征无前向优化空间，非 skill/agent 失败）· **厂商库墙区**（前向 baseline 就是 NVIDIA 厂商成品如 cholesky=cuSOLVER——手写分块极难赢厂商库，认边界+手写尽力+诚实报）。**带宽墙/厂商库墙的反向常仍能赢**（有 Jacobian/散回/解析 Φ 算子/伴随系统等结构）。

**核心结论**：
- **能否赢 torch.compile 的元判据**：candidate 能赢当且仅当做到其一——① 访存比 baseline 更少（融合省中间物化/避免多趟）；② 用了 autograd/torch.compile 不知道的解析数学结构（伴随系统/Φ 算子/稳定递推，反向常数量级优势，如 tridiag 反 9.80×、cholesky 反 1.25×即便前向输厂商库）；③ 算术强度够高让计算而非访存主导（大 reduce/点积/GEMM 融合尾）。三者都不占（纯访存前向/厂商库前向）→ 打不过，认边界。**反向常比前向好赢**（前向多是搬运=带宽墙，反向含 Jacobian/伴随/多超越函数=可融合真赢）。
- **三维独立**：skill 质量 / 模型能力 / 宿主架构自主性互不相同，测试须三维分离归因（同一 case codex 稳过、aider/gptme 可能因架构或能力栽）。
- **弱 baseline 5 变种**（reference 不诚实致加速比虚高）：① Python for 循环 ② O(N²) Toeplitz 伪向量化 ③ 规模专属慢分支 ④ cumprod/cumsum 数值脆弱+反向畸形 ⑤ 用更慢通用算子(sort/dense linalg.solve)替代最优原语(topk/PCR)。均由 [`skill/scripts/check_reference.py`](skill/scripts/check_reference.py) 静态预检自动预警。
- **评测鲁棒性加固**：`run_on_a100.sh --auto-scale` 自适应放大到"计算主导区"（补短核假象）+ 规模敏感复测（擦线 PASS 放大掉破 1.05 判 `PASS_SCALE_SUSPECT`）+ `check_reference.py` 静态扫危险写法（补弱 baseline + kernel 嵌套重算 + 直调厂商库成品）+ 防作弊红线 §1-5 + 擦线 3 连稳定判据 + 驱动器健壮性（tar 自动回退 System32 bsdtar 绕杀软拦截 + 远程执行墙钟超时防 WSL 冻结）。
- **红线准则**：torch 高层算子（`F.*`/`nn.*`/`matmul`/`sparse.mm`/SDPA）禁；CUDA 官方底层库（cuBLAS/CUB/cuSPARSE/cuSOLVER）**仅作辅助原语**（GEMM/TRSM/scan 积木自己拼算法）允许，**禁直调与目标算子等价的库成品**（Cholesky 直调 `cusolverDnSpotrf`、解线性系统 `getrf/gesv`、FFT cuFFT——那 candidate=baseline 同款厂商算法，失去手写跑赢意义）；通用张量原语（`topk`/`sort`/`cumsum`/`scatter_add`）reference 里允许但 candidate 须手写 `.cu`。



两层架构：**通用框架**（算法无关）+ **算法 case**（每算法一份，可替换）。RBF 是第一个 case，后续加新算法只需新增 `cases/<name>/`。

| 路径 | 用途 |
|------|------|
| `framework/` | 通用框架（**算法无关**）：`protocol.py` 计时/容差协议、`case.py` Case 协议、`verify.py` 正确性验证、`bench.py` 计时、`loader.py` 编译加载、`smoke.cu` 冒烟。**评测脚本对 agent 只读、进程隔离** |
| `cases/<name>/` | 算法 case：`__init__.py`(暴露 CASE)、`reference.py` 金标准、`config.py` 形状/参数、`description.md` NL 描述、`kernels/` CUDA kernel、`op.py` autograd 封装；`delivery/` 可独立编译的纯 CUDA 交付版 |
| `skill/` | 交付的 skill：`DESIGN.md` 设计、`SKILL.md` 方法论、`loop.md` 迭代循环、`scripts/` 通用 CLI（`verify_case.py`/`bench_case.py`/`profile_case.py --case <name>`） |
| `notebooks/` | Colab 驱动 notebook（clone/pull + 装依赖 + 运行入口） |
| `scripts/` | 通用脚本（如 `probe_env.py` 环境探测） |

已验证的 case（**23 形态**，覆盖归约/矩阵乘/scan/卷积/位置编码/数据依赖写/间接寻址/点积/分组归约/逐元素融合链/空间采样/变长分段/线性求解/稠密分解等维度，framework 零改动即支持每一个）：`rbf` · `layernorm` · `softmax_ce` · `scan` · `gemm_bias_gelu` · `online_softmax`（§12 有实测，未 collect 成 case 目录）· `rope` · `linear_ssm` · `welford` · `causal_attn` · `conv1d` · `gated_ssm` · `scatter_add` · `topk` · `maxpool` · `spmv` · `cosine_sim` · `temperature_softmax` · `l2norm_scale` · `groupnorm` · `geglu` · `gridsample` · `segment_softmax` · `tridiag` · `cholesky`。三宿主（aider/gptme/codex）逐形态对照与结论见 [`skill/AGENT_TEST_MATRIX.md`](skill/AGENT_TEST_MATRIX.md)。

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
