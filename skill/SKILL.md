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

## Case 协议（必读——接口写错会直接导致 verify 崩溃）

`framework/case.py` 的 `Case` 是一个 dataclass，**7 个字段全部必填**（顺序无关，用关键字传参）。不要猜、不要写 try/except 兼容——照下面模板填：

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | str | case 名，如 `"softmax_ce"` |
| `description` | str | 自然语言描述（通常读 `description.md`） |
| `params` | dict | 算法标量参数，如 `{"gamma": 1/64}`；无则 `{}` |
| `grad_inputs` | list[str] | 需要求梯度的**输入名**，如 `["logits"]`、`["X","Y"]` |
| `dtype` | str | 验收精度，如 `"float32"` |
| `make_inputs` | callable | `(seed, dtype, device, requires_grad) -> dict[str,Tensor]`（**只返回 dict 一个值**，不要返回 `(dict, params)` 二元组！） |
| `reference_forward` | callable | `(inputs: dict, params: dict) -> output_tensor` |

**关键约定**：
- **输入是 `dict`（不是 tuple！）**：`make_inputs` 返回 `{"名字": 张量, ...}`；`reference_forward` 用 `inputs["名字"]` 取。framework 的 verify/bench 按 dict 传参并按 `grad_inputs` 里的名字取梯度。
- **`make_inputs` 只 `return inputs`（那个 dict）**——**不要** `return inputs, params`。params 已在 `CASE` 里单独提供，从 make_inputs 返回二元组会让 verify 收到 tuple 而报 `tuple indices must be integers`。
- 不求梯度的输入（如整型 labels）也放进 dict，但**不列进 `grad_inputs`**，且不受 `dtype` 影响（labels 恒 int64）。
- 输出可以是标量（如 loss）或张量；upstream 梯度由 framework 按输出形状自动生成。

**可直接照抄的 `cases/<name>/__init__.py` 骨架**：
```python
import os
from framework.case import Case
from cases.<name> import config
from cases.<name>.reference import reference_forward, make_inputs

def _load_description():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(p, encoding="utf-8") as f:
        return f.read()

CASE = Case(
    name="<name>",
    description=_load_description(),
    params={},                    # 或 {"gamma": ...}
    grad_inputs=["<输入名>"],      # 需求梯度的输入
    dtype="float32",
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
```

**可直接照抄的 `cases/<name>/reference.py` 骨架**（注意 `make_inputs` 只 `return inputs`）：
```python
import torch
from cases.<name> import config

def reference_forward(inputs, params):
    x = inputs["<输入名>"]              # 用 dict 取，不是 tuple 解包
    # ... 用基础算子表达前向，禁止落回 F.* 高层算子 ...
    return out                          # 标量或张量

def make_inputs(seed, dtype, device, requires_grad=False):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(config.N, config.D, dtype=dtype, device=device, generator=g)
    if requires_grad:
        x.requires_grad_(True)          # 只对 grad_inputs 里的输入置 True
    return {"<输入名>": x}              # 只返回 dict！不要 return inputs, params
```

**`op.py` 候选契约**：`candidate(inputs: dict, params: dict) -> output`，用 `torch.autograd.Function` 包装前反向 kernel，对 `grad_inputs` 中的输入返回梯度、其余返回 `None`。（照抄 `cases/rbf/op.py`。）

> 拿不准接口时，直接读 `framework/case.py`（就几十行）和 `cases/rbf/__init__.py`——不要凭猜写。

## 工作流程

### 步骤 0：确认环境
在带 NVIDIA GPU 的机器上运行（本项目用 Colab T4，sm_75）。先跑 `python framework/smoke_test.py` 确认编译链路（nvcc + ninja）可用。

### 步骤 1：解析算法描述
从 NL 描述 + shape/dtype 明确：**输入张量**（名字、形状）、**输出**、**标量参数**、**需要求梯度的输入**（`grad_inputs`）。这些决定 Case 的字段。

### 步骤 2：写 PyTorch 参考实现（金标准）
在 `cases/<name>/reference.py` 用**基础算子**（广播、matmul、逐元素、规约）表达前向，autograd 自动提供反向。
- **红线**：待测路径禁止落回 `F.scaled_dot_product_attention` / `torch.nn.functional` 等高层融合算子。
- 用自然、直接的写法翻译描述（不要为了让 baseline 变慢而扭曲，也不要用 GEMM 分解等把 baseline 推向 cuBLAS 而变得打不过）。
- **必须向量化，禁止 Python 沿任何张量维度的 for 循环**（**包括时序/序列维度**，不只是"逐元素"或"逐列"）：算法描述里的"逐步/单遍/在线扫描/逐列累加/沿时序递推"等**串行叙述是数学语义，不是实现方式**——要用张量广播+整体规约（如 `logits.amax(-1,keepdim=True)`、`exp(...).sum(-1)`、`torch.cumsum`、`torch.pow(a, arange(T))` 等）等价表达，**禁止 `for t in range(T):`/`for c in range(C):` 等任何维度的迭代**。原因有二：①`framework/bench.py` 会对 reference 做 `torch.compile` 作 baseline，Python 循环会被展开成 O(N) 巨型图，**首次编译几十秒甚至卡死 bench**（online_softmax 曾因逐列循环 C=1024 → torch.compile 编译 44s 卡挂；linear_ssm gptme 版因 `for t in range(1024)` → bench 死锁 fork 进程炸弹）；②逐元素循环的 eager baseline 畸形慢，会造成**"弱 baseline 假象"**——candidate 跟畸形慢的基线比，加速比虚高但不诚实（类比[短核假象]）。**自检**：reference 里出现 `for` 遍历任何张量维度（B/T/H/C 等），几乎一定写错了，改成向量化算子。
- **向量化必须算法复杂度正确，禁止 O(N²) 密集矩阵代替最优 O(N) scan/cumsum**：把线性递推/scan 类算法（如 `h_t = a*h_{t-1} + b*x_t`）表达成 `T×T` 下三角 Toeplitz 密集矩阵 + `einsum`/`matmul` **是错的向量化**——它算法复杂度从 O(T) 恶化到 O(T²)，baseline `torch.compile` 会做 T² 次乘加（不利用三角结构），是**弱 baseline 假象的高级变种**（不是慢在编译，而是慢在算力）。linear_ssm gptme 曾用 `torch.tril(W) + einsum` 拿到 44×/54× 虚高加速比（真实 O(T) baseline 下 codex/aider 只有 3~4×）。**scan/递推类算法的正解**：用数学变换转成 `cumsum`（如 `h_t = a*h_{t-1}+b*x_t` ⟹ `h_t = b·a^t·cumsum(x/a^t)[t]`）、`cumprod`、`cummax` 等 O(N) 前缀原语；PyTorch 已优化好，torch.compile 会用 GPU 并行 scan。**自检**：reference 里出现 `T×T`/`N×N` 中间矩阵（含 `torch.tril`/`triu`/`einsum('tk,...')`）而算法本可 O(N) 表达，一定是错的向量化。
  - **例外——真变系数递推（数学上无 O(N) 数值稳定向量化时，O(T²) 下三角合法且诚实）**：当递推的系数**输入依赖**（每步系数随输入变化，如门控 SSM/Mamba 类 `h_t = z_t·h_{t-1}+(1-z_t)·x_t`、`z_t` 依赖 `x_t`），且**不存在数值稳定的 O(N) 前缀形式**时（判断依据：把它写成 `cumprod`/`cumsum` 的 O(N) 形式会因系数连乘而下溢/上溢导致 NaN，或其 autograd 反向图畸形慢；而 `for` 循环又会让 torch.compile 编译爆炸），**O(T²) 下三角权重矩阵是合法、诚实的 reference**——因为它是该算法**数值稳定的最优可行 baseline**，不属于"偷懒用 T² 代替本可 O(N) 的"。**区分准则**：系数与输入无关的固定递推（如 linear_ssm）→ 有稳定 O(N) 前缀形式 → 禁 T²、必须 cumsum；系数输入依赖的变系数递推 → 若无稳定 O(N) → O(T²) 下三角合法。**自主判断**：先尝试 O(N) 前缀形式，若数值实验（verify）发现 NaN/溢出或反向异常慢，再退到数值稳定的 O(T²) 下三角，并在 reference 注释说明为何 O(N) 不可行。
- **禁止用 `make_inputs` 挑异常输入分布迁就脆弱 reference**（评测作弊点）：`make_inputs` 的随机分布应当**自然、有代表性**（如标准正态、常规参数范围），**禁止**为了让数值脆弱的 reference 不 NaN/不溢出而挑异常分布——例如让某个 sigmoid/exp 激活因偏置或缩放被设成恒接近饱和值（0 或 1），从而让脆弱的 `cumprod`/`cumsum` 链不下溢。这是"把 reference 的数值缺陷藏进输入分布"。**自检**：make_inputs 里激活的偏置/缩放被设成让其恒接近饱和、或参数范围明显偏离常规，警惕是在迁就 reference——应改用自然分布，若此时 reference 数值崩，说明 reference 写法本身有问题（见上条"真变系数递推"的处理）。
- **reference 必须是单一、最干净的向量化路径，禁止规模/条件专属分支**：`reference_forward` 里**禁止**写 `if x.numel() >= 阈值: <快路径> else: <慢路径>` 这类按输入规模/形状切换实现的分支——金标准应当**始终**用同一份最优向量化（如 LayerNorm 恒用 `x.mean(-1)`+`((x-mean)**2).mean(-1)`）。**危害**：若"小张量分支"用了更慢的写法（如 `cumsum`+一串中间张量），而 bench 默认规模恰好命中它，baseline 就被人为拖慢 → **弱 baseline 假象**（welford aider 版曾因 `numel<64M 走 cumsum 慢分支`、bench 默认 16.7M 命中 → 刷出前2.71/反2.89× 虚高，强制走干净分支后真实仅 0.99/0.91×）。这种陷阱比 for/T² 更隐蔽（reference 表面无 for/tril，grep 查不出）。**自检**：reference 里出现按 `numel`/`shape`/`size` 分支选实现，删掉分支只留最干净那条；不同规模下 candidate 加速比若差异巨大（如小规模 2.7×、大规模 0.99×），几乎一定是 reference 藏了规模专属慢路径。
- 在 `config.py` 固定 shape/参数，在 `__init__.py` 组装 `CASE`（**严格照上面"Case 协议"章节的 7 个必填字段和骨架模板**，别猜；`cases/rbf/` 是完整范例）。
- **可自查（reference 静态预检）**：写完 reference 可跑 `python skill/scripts/check_reference.py --case <name>` 静态扫上述弱 baseline 危险写法（for 遍历张量维 / tril+einsum 的 O(T²) 伪向量化 / numel-shape 规模分支 / cumprod+除法数值脆弱 / F.* 高层算子落回 / make_inputs 挑异常输入分布）。输出 `REF_CHECK=CLEAN` 表示无可疑；命中则打印 WARN 供核查（有合法例外如变系数递推 T²，故只警告不拦）。`run_on_a100.sh` 每次自测前也会自动跑此预检并把 WARN 打进日志——见到 WARN 先核查是不是弱 baseline。

### 步骤 3：写 CUDA 前向 kernel → 验证
在 `cases/<name>/kernels/` 写前向 `.cu`，`op.py` 里用扩展加载并提供 `forward_only`/`candidate`。运行：
```
python skill/scripts/verify_case.py --case <name>
```
看前向 5 种子是否全 PASS（allclose atol=rtol=1e-2）。失败则读 `max_abs_err` 判断是索引/边界/数值问题。

### 步骤 4：写 CUDA 反向 kernel → 验证
在 `kernels/` 写反向 `.cu`（计算各 `grad_inputs` 的梯度），`op.py` 用 `torch.autograd.Function` 把前反向包成 `candidate(inputs, params)`。再跑 `verify_case.py`，确认反向各梯度也全 PASS。
- **反向数学自主获取**（不依赖 description 给公式）：PyTorch 参考的反向本就由 autograd 自动提供，所以**金标准梯度是现成的**——你只需让 CUDA 反向 kernel 的数值**复现 autograd 的梯度**，用 `verify_case.py` 对拍校验即可。若需手推，用链式法则通用套路（先 `dL/d(中间量)`，再链到各输入），见下方"CUDA Kernel 实现技巧"。

### 步骤 5：计时对比 torch.compile
```
python skill/scripts/bench_case.py --case <name>
```
看前向、反向各自加速比是否 ≥1.05×。CV>5% 表示测量噪声、结果作废需重测（非 kernel 问题）。

### 步骤 6：未达标 → 进入优化循环
若正确但未达速度标，按 [loop.md](loop.md) 迭代：读 profile → 优化（shared-memory tiling / float4 向量化 / warp 原语 / 算子融合 / 前向缓存复用）→ 重新 verify（必须仍全 PASS）→ 重新 bench。每次只改 kernel，不动 framework。

### 步骤 7：交付
产出可独立编译的 `.cu`（含必要 host 绑定），确认无对 torch 高层算子的运行时依赖。

## CUDA Kernel 实现技巧（提前学习——用户只给朴素算法描述，这些知识靠你自己掌握）

用户的 description 通常只有自然语言 + 前向公式 + shape，**不会给反向公式、数值稳定或优化提示**。
以下是实现任意算法 kernel 都通用的技巧，据此自主完成，不要指望用户在输入里喂。

### 瓶颈诊断 → 策略选择（优化时先做这一步，别顺着清单盲试）

**优化不达标的 kernel 时，先用 `profile_case.py` 诊断瓶颈类型，再据下表直接选手段——而非从头试每一招。**
瓶颈类型有限、可枚举，故本表对**任意算法（含全新的）通用**：先把 kernel 归到某类瓶颈，再上对应手段。

| profile 观察到的信号 | 瓶颈类型 | 优先手段（→ 下面 A-F 小节） |
|---------------------|---------|--------------------------|
| kernel 本体时间 << 墙钟；`aten::empty`/`cudaMalloc`/launch 占比高 | 固定开销 | 融合减 kernel 数、减 launch；短核先放大规模测（bench.env）(F) |
| 线程数 < SM 可驻留量（"一线程一行/一输出"）| occupancy 低 | 降 block 到 256、thread coarsening (E) |
| 全局访存量 >> 理论下界；同 warp 跨大步长读 | 访存未合并/重复读 | 合并访问、shared tiling、**float4 向量化 + 寄存器缓存**(C/D) |
| 反向重算了前向已算过的中间量（mean/std、softmax、K…）| 重算 | **前向 `ctx` 缓存复用**(F)——反向决定性一招 |
| 规约（sum/mean/max）占大头 | 规约低效 | warp shuffle 规约、两级归约、**列规约二维分块**(C) |
| 朴素参考物化 [N,M,D] 等大张量 | 带宽/内存 | 融合不物化 (F) |
| 算法是**累积/扫描依赖**（前缀和、cumsum、cumprod、扫描类）| 串行依赖 | 用成熟**并行扫描原语**（CUB `BlockScan`/`DeviceScan`，或 Hillis-Steele/Blelloch）；反向是**反向扫描**（`dx[j]=Σ_{i≥j} dy[i]`）。实测 scan 用 CUB block scan 前3.4×/反4.1× 大幅达标 |

> **实测印证（LayerNorm 反向，曾被误判为"reduce 密集打不过 torch.compile 的天花板"）**：按上表诊断后三招组合即稳过 5%——
> ①反向**缓存 mean/rstd**（前向输出、反向读，消除朴素 O(B·D²) 重算）反向 ~1.0×→1.11×；②dgamma/dbeta **二维分块列规约**；
> ③前向 **float4+寄存器缓存**（X 从三遍读降到一遍）前向 0.93×→1.09×。**"打不过"往往是没诊断对瓶颈、没上缓存复用，不是真天花板。**

### A. 反向梯度：不用等用户给公式
- **autograd 即金标准**：写对 PyTorch 前向后，`verify_case.py` 会用 autograd 自动算出参考梯度。你的 CUDA 反向 kernel 只要**数值上复现它**即可——对拍着调，误差进 atol 就对了。
- **手推套路**（需要时）：设中间量（如 dist、softmax、mean/std），先求 `dL/d(中间量)=上游梯度·d(输出)/d(中间量)`，再链式到各输入。逐元素运算梯度就地乘；规约（sum/mean/max）的反向是"广播回去"；矩阵乘 `C=A@B` 的反向是 `dA=dC@Bᵀ, dB=Aᵀ@dC`。
- 只对 `grad_inputs` 里的输入返回梯度，其余（如整型 labels）在 autograd.Function.backward 里返回 `None`。

### B. 数值稳定（自己就要想到）
- **softmax/logsumexp**：先减去该行 max 再 exp（`exp(x-max)`），避免上溢；`logsumexp = max + log(Σexp(x-max))`。
- **除法/归一化**：分母加 `eps`（如 `1/sqrt(var+eps)`）。
- 避免大数相消；累加用 float 累加器（即便输出 fp32，规约中间量也别过早截断）。

### C. 规约模式（mean/var/sum/max/softmax 都用得上）
- **block-per-row（行规约）**：一个 block 处理一行/一个样本，block 内多线程 grid-stride 遍历该行，再树形规约（`shared[t]+=shared[t+s]`，`s` 折半，配 `__syncthreads()`）。行主序下这条访存合并、最常用。
- **warp 规约**：一个 warp 内用 `__shfl_down_sync` 免 shared 往返，更快。
- **两遍规约**：如 var 需先得 mean 再求平方差和；logsumexp 需先 max 再 sum。行内规约做两趟即可。
- **列规约（沿行数 B 规约，如 LayerNorm 的 dgamma/dbeta = Σ_b …）——别用"一 block 一列"的朴素写法**：那样同一 warp 的线程读**同列不同行**（跨 B×4 字节大跨步），访存完全不合并，且只有 D 个 block、并行度低，是常见慢点。改用二维分块：
  - **网格按 (行块 × 列)** 布置，每个 block 覆盖一段行 × 一片连续列，block 内线程按**行主序合并读**（`threadIdx.x` 对应列、连续），在寄存器/shared 里对本行块做部分和；
  - 多个行块的部分和再**跨 block 累加**（`atomicAdd` 到输出，或第二个 kernel 做行块间归约）。这样既合并访存、又有足够 block 喂满 SM。
  - 简化可行版：仍一 block 一列但让 block 内线程沿 B 分段、每线程累加多行再 block 内树形规约——比一线程扫全列强，但仍不如二维分块合并访存。

### D. 访存优化
- **float4 向量化**：连续维度按 `float4`（或 `reinterpret_cast<const float4*>`）一次读 4 个，减少 load 指令、提升带宽利用。要求 16B 对齐、维度是 4 的倍数。
- **合并访问**：同一 warp 的线程访问连续地址；行主序下让 `threadIdx.x` 对应最内维。
- 指针加 `__restrict__` 帮编译器优化。

### E. occupancy 与并行度
- **别用过大 block**：如 1024 线程/block 常使每 SM 只驻留 1 个 block、占用率仅 ~50%。256 线程/block 通常更优（可多 block 并存）。
- **thread coarsening**：每线程算多个输出（如 2×2 微块），减少总 block 数、复用寄存器里的数据。
- 并行度不足（如"一线程算一整行"只有 N 个线程）是常见慢因——让线程数足够喂满所有 SM。

### F. 避免重算 / 融合
- **前向缓存复用**：反向要用的前向中间量（K、softmax、mean/std 等），在 `autograd.Function.forward` 里 `ctx.save_for_backward` 存下，反向直接读，别重算。这常是反向提速的决定性一招。
- **算子融合**：把多步逐元素/规约（如 距离→exp、norm→仿射）融进一个 kernel，避免中间张量物化和额外 kernel launch。
- **避免物化大中间量**：若朴素参考会物化 [N,M,D] 之类的大张量（广播式），手写融合 kernel 全程不物化即是内存带宽优势来源。

> 优化时不必一次全上——先写朴素正确版过 verify，再按 [loop.md] 按收益逐项加（先提 occupancy，再缓存复用/融合，再 float4/warp）。每次改完先 verify 再 bench。

## 防作弊红线（不可违反）

1. 待测路径禁止落回 `F.scaled_dot_product_attention` / `torch.nn.functional` / `torch.matmul` / `torch.sparse.mm` 等 **torch 高层算子**。
   **但允许 CUDA 官方底层库**（cuBLAS `cublasSgemm`、CUB `BlockScan/DeviceScan`、cuSPARSE 等）——它们是 CUDA 生态的底层原语、
   非 torch 高层封装，与本红线不冲突。判据：candidate 走的是自定义 `.cu`（可在其中调 cuBLAS/CUB/cuSPARSE + 手写融合/逐元素 kernel），
   不是回到 PyTorch 的高层张量算子。实测判例：GEMM+bias+gelu 用 `cublasSgemm` 做矩阵乘 + 手写 bias+gelu 融合尾 → 合规
   （赢 torch.compile 靠融合省中间物化/额外 launch，是正当优势）；scan 用 CUB block scan → 合规；SpMV 用 cuSPARSE 或手写 CSR kernel → 合规
   （注意 cuSPARSE 通用路径有 descriptor/workspace 固定开销，反向常因此打不过 torch.compile 的融合 scatter——手写融合 kernel 通常更优）。
   **通用张量原语**（`torch.topk`/`torch.sort`/`torch.cumsum`/`torch.scatter_add`/`torch.index_select` 等）**在 reference 里允许**（它们是基础操作、非神经网络高层层算子），但 candidate 仍须手写 `.cu` 不得直接调这些 torch 原语糊弄；**神经网络层算子**（`F.*`/`nn.*`：max_pool/layer_norm/conv/sdpa/embedding 等）**reference 也禁**。
2. 禁止修改/绕过 `framework/` 下的验证器、计时器、协议（评测基座只读）。
3. 禁止降精度换速度（除非算法描述本身指定低精度）。
4. 交付 `.cu` 须能独立编译、无 torch 高层运行时依赖。
5. **评测路径必须等于真实路径——禁止计时特化**：`op.py` 的 `candidate`/前向不得针对评测的测量方式（如 bench 用
   `torch.no_grad()`+`detach()` 计前向）走一条真实使用时不会走的"快路径"。典型违规：检测到输入无 `requires_grad`
   就绕过 `autograd.Function`、跳过反向所需中间量（mean/rstd/K 等）的保存——这只在计时时受益，带反向的真实前向拿不到，
   属钻空子。**candidate 必须始终返回可反向的、与 verify 同一份实现的输出**；前向优化只能来自 kernel 本身，不能来自
   "计时时少干活"。—— 实测教训：codex 曾在 op.py 加"无梯度绕过 autograd + 跳过 mean/rstd 存储"分支，把前向从诚实
   1.04× 刷到 1.08×，撤销该特化后即塌回 1.04×FAIL。

## 达标判据

- 正确性：≥5 组随机输入，前向 + 每个 `grad_inputs` 的反向梯度均 `allclose(atol=rtol=1e-2)`。
- 性能：前向、反向各自相对 `torch.compile`（默认 mode）≥1.05×，3 次重跑 CV≤5%。
- **稳定过线**：加速比擦线（1.05–1.10× 区间）时，单次 PASS 不算达标——须连跑 3 次全 PASS 才算真达标；
  共享/繁忙 GPU 上擦线加速比会在达标线上下抖动，达标应留安全余量（目标 ≥1.10×）而非骑在 1.05 线。
- **警惕短核假象**：若 baseline 前/反向 <0.15ms 却给高加速比（1.2×+），多半是固定开销虚高。**只要 config 的规模支持 env 覆盖，
  `run_on_a100.sh` 会自动探测短核并放大规模重测（harness 兜底，无需你建 bench.env）**——所以务必让 config 参数化规模
  （`os.environ.get("XXX", "默认")`）。可选：短核 case 建 `bench.env` 声明规模更明确。实测：aider 的 RMSNorm 短核下显示 前1.22/反1.43，
  harness 自动放大后真实 前0.86/反1.00 FAIL——兜底让不建 bench.env 的 agent 也不被短核假象骗。
- 两者同时满足即达成。

## 新增一个算法 case

复制 `cases/rbf/` 为 `cases/<name>/`，替换 `reference.py`（新算法的金标准）、`config.py`（新形状/参数）、`description.md`（新描述）、`__init__.py` 的 `CASE`（`grad_inputs`/`params`/`name`），清空 `kernels/` 重写。framework 与 CLI 无需改动，`--case <name>` 即可复用整套验证/计时。
