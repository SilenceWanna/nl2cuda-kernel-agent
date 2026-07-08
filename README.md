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

## 目录结构

两层架构：**通用框架**（算法无关）+ **算法 case**（每算法一份，可替换）。RBF 是第一个 case，后续加新算法只需新增 `cases/<name>/`。

| 路径 | 用途 |
|------|------|
| `framework/` | 通用框架（**算法无关**）：`protocol.py` 计时/容差协议、`case.py` Case 协议、`verify.py` 正确性验证、`bench.py` 计时、`loader.py` 编译加载、`smoke.cu` 冒烟。**评测脚本对 agent 只读、进程隔离** |
| `cases/<name>/` | 算法 case：`__init__.py`(暴露 CASE)、`reference.py` 金标准、`config.py` 形状/参数、`description.md` NL 描述、`kernels/` CUDA kernel、`op.py` autograd 封装；`delivery/` 可独立编译的纯 CUDA 交付版 |
| `skill/` | 交付的 skill：`DESIGN.md` 设计、`SKILL.md` 方法论、`loop.md` 迭代循环、`scripts/` 通用 CLI（`verify_case.py`/`bench_case.py`/`profile_case.py --case <name>`） |
| `notebooks/` | Colab 驱动 notebook（clone/pull + 装依赖 + 运行入口） |
| `scripts/` | 通用脚本（如 `probe_env.py` 环境探测） |

已验证的 case：**rbf**（成对距离/RBF 核，验收用例）、**layernorm**（层归一化，验证通用性——单输入+参数张量+标量、3 个待求梯度，framework 零改动即支持）。

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
