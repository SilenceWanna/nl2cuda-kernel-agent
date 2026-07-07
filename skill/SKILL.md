---
name: nl2cuda-kernel
description: 把自然语言描述的算法结构（含 shape/dtype）自动实现为自定义 CUDA 前向+反向 kernel，以 PyTorch 参考实现为金标准通过正确性验证，并在规范计时下超过 torch.compile。当用户要求"为某算法写/生成/优化 CUDA kernel"、"实现前反向 kernel 并跑赢 torch.compile"、"把这个算法结构变成 .cu"时使用。
---

# 自然语言 → CUDA 前反向 Kernel 生成

把用户用自然语言描述的算法结构，自动实现为可独立编译的自定义 CUDA 前向+反向 kernel：以 PyTorch 参考实现为正确性金标准，通过 ≥5 组随机输入的前反向 allclose，并在规范计时下让前向与反向都比 `torch.compile`（默认 mode）快 ≥5%。

本 skill **算法无关**：RBF 高斯核矩阵只是内置的第一个 case，同样流程适用于任意算法结构（归一化、损失、距离、扫描等）。

## 何时用

- 用户给出一个算法的自然语言描述 + 张量 shape/dtype，要求生成 CUDA 前反向 kernel。
- 用户要求把已有算法"手写成 CUDA kernel 并跑赢 torch.compile"。
- 用户要求为新算法结构新增一个 case 并走完验收。

## 架构（先读 [DESIGN.md](DESIGN.md)）

两层：
- `framework/`（**算法无关，对你只读**）：`protocol.py` 计时/容差协议、`case.py` Case 协议、`verify.py` 正确性验证、`bench.py` 计时、`loader.py` 编译加载、`smoke.cu` 冒烟。
- `cases/<name>/`（每算法一份，**你在这里写代码**）：`reference.py` 金标准、`config.py` 形状/参数、`description.md` NL 描述、`kernels/*.cu` CUDA kernel、`op.py` autograd 封装、`__init__.py` 暴露 `CASE`。

## 工作流程

### 步骤 0：确认环境
在带 NVIDIA GPU 的机器上运行（本项目用 Colab T4，sm_75）。先跑 `python framework/smoke_test.py` 确认编译链路（nvcc + ninja）可用。

### 步骤 1：解析算法描述
从 NL 描述 + shape/dtype 明确：**输入张量**（名字、形状）、**输出**、**标量参数**、**需要求梯度的输入**（`grad_inputs`）。这些决定 Case 的字段。

### 步骤 2：写 PyTorch 参考实现（金标准）
在 `cases/<name>/reference.py` 用**基础算子**（广播、matmul、逐元素、规约）表达前向，autograd 自动提供反向。
- **红线**：待测路径禁止落回 `F.scaled_dot_product_attention` / `torch.nn.functional` 等高层融合算子。
- 用自然、直接的写法翻译描述（不要为了让 baseline 变慢而扭曲，也不要用 GEMM 分解等把 baseline 推向 cuBLAS 而变得打不过）。
- 在 `config.py` 固定 shape/参数，在 `__init__.py` 组装 `CASE`（见 `cases/rbf/` 为模板）。

### 步骤 3：写 CUDA 前向 kernel → 验证
在 `cases/<name>/kernels/` 写前向 `.cu`，`op.py` 里用扩展加载并提供 `forward_only`/`candidate`。运行：
```
python skill/scripts/verify_case.py --case <name>
```
看前向 5 种子是否全 PASS（allclose atol=rtol=1e-2）。失败则读 `max_abs_err` 判断是索引/边界/数值问题。

### 步骤 4：写 CUDA 反向 kernel → 验证
在 `kernels/` 写反向 `.cu`（计算各 `grad_inputs` 的梯度），`op.py` 用 `torch.autograd.Function` 把前反向包成 `candidate(inputs, params)`。再跑 `verify_case.py`，确认反向各梯度也全 PASS。
- 反向数学：先手推 `dL/d(中间量)`，再链式到各输入梯度（RBF 的推导见 `cases/rbf/description.md`）。

### 步骤 5：计时对比 torch.compile
```
python skill/scripts/bench_case.py --case <name>
```
看前向、反向各自加速比是否 ≥1.05×。CV>5% 表示测量噪声、结果作废需重测（非 kernel 问题）。

### 步骤 6：未达标 → 进入优化循环
若正确但未达速度标，按 [loop.md](loop.md) 迭代：读 profile → 优化（shared-memory tiling / float4 向量化 / warp 原语 / 算子融合 / 前向缓存复用）→ 重新 verify（必须仍全 PASS）→ 重新 bench。每次只改 kernel，不动 framework。

### 步骤 7：交付
产出可独立编译的 `.cu`（含必要 host 绑定），确认无对 torch 高层算子的运行时依赖。

## 防作弊红线（不可违反）

1. 待测路径禁止落回 `F.scaled_dot_product_attention` / `torch.nn.functional` 等高层算子。
2. 禁止修改/绕过 `framework/` 下的验证器、计时器、协议（评测基座只读）。
3. 禁止降精度换速度（除非算法描述本身指定低精度）。
4. 交付 `.cu` 须能独立编译、无 torch 高层运行时依赖。

## 达标判据

- 正确性：≥5 组随机输入，前向 + 每个 `grad_inputs` 的反向梯度均 `allclose(atol=rtol=1e-2)`。
- 性能：前向、反向各自相对 `torch.compile`（默认 mode）≥1.05×，3 次重跑 CV≤5%。
- 两者同时满足即达成。

## 新增一个算法 case

复制 `cases/rbf/` 为 `cases/<name>/`，替换 `reference.py`（新算法的金标准）、`config.py`（新形状/参数）、`description.md`（新描述）、`__init__.py` 的 `CASE`（`grad_inputs`/`params`/`name`），清空 `kernels/` 重写。framework 与 CLI 无需改动，`--case <name>` 即可复用整套验证/计时。
