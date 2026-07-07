# nl2cuda-kernel-agent

自然语言 → CUDA 前反向 Kernel 自动生成 **Skill**。

用户用自然语言描述一个算法结构（附张量 shape/dtype 约定），挂载了本 skill 的现成 agent（Claude Code / Codex / 开源 agent）自动完成闭环：

1. **PyTorch 参考实现** —— 由自然语言生成可运行的前反向参考，作为正确性金标准；
2. **CUDA Kernel 生成** —— 生成自定义前向 + 反向 kernel；
3. **优化迭代** —— 基于 profile 反馈做 tiling / float4 / warp 原语 / 算子融合等优化；
4. **交付** —— 产出可独立编译的 `.cu` 文件。

> 交付物是 **skill（方法论 + 工具）**，不是从零构建的 agent。详见 [工作目标.md](工作目标.md)（任务契约）与 [工作计划.md](工作计划.md)（执行清单）。

## 验收用例

**成对距离 / RBF 高斯核矩阵**（非 attention），FP32，形状 **N=M=4096, D=64**：

- 前向：`K[i,j] = exp(-gamma · ||x_i − y_j||²)`，X:[N,D], Y:[M,D] → K:[N,M]
- 反向：对 X、Y 求梯度 dX、dY

**验收标准**：以 PyTorch 参考为金标准，≥5 组随机输入前反向均 `allclose(atol=rtol=1e-2)`；且自定义 CUDA kernel 相对 `torch.compile`（默认 mode）前向与反向均快 ≥5%。

## 目录结构

| 路径 | 用途 |
|------|------|
| `reference/` | PyTorch 参考实现 + 验收用例配置（正确性金标准） |
| `tests/` | 正确性验证器（allclose + gradcheck） |
| `benchmarks/` | 计时基准器（CUDA events，对比 torch.compile）；**只读、进程隔离** |
| `kernels/` | 手写 CUDA kernel（管线自检 + skill 参考解）与编译加载脚手架 |
| `skill/` | 交付的 skill：方法论 `SKILL.md` + 工具 `scripts/` + loop 驱动 |
| `notebooks/` | Colab 驱动 notebook（clone/pull + 装依赖 + 运行入口） |
| `scripts/` | 通用脚本（如 `probe_env.py` 环境探测） |

## 开发环境

- 开发机无 NVIDIA GPU，仅写代码 + git。
- GPU 编译/运行/验收在 **Google Colab**（免费版 Tesla T4，sm_75，nvcc 12.8）上进行。
- 本仓库为 public，Colab 里 `git clone` 无需认证。

## 快速开始（在 Colab）

```bash
!git clone https://github.com/SilenceWanna/nl2cuda-kernel-agent.git
%cd nl2cuda-kernel-agent
!python scripts/probe_env.py   # 确认 GPU / CUDA / PyTorch
```

后续验证与基准入口见 `notebooks/run.ipynb`。
