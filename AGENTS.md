# AGENTS.md — 本仓库的 agent 工作约定（自动加载）

> codex 会自动读本文件；aider 读 `CONVENTIONS.md`（内容相同，指向本文件）；Claude Code 读 `CLAUDE.md`。
> **目的**：用户在交互界面**只需输入算法定义**，你就按本约定 + `skill/SKILL.md` 自动走完全流程，
> 无需用户再手输"读 SKILL、自主推导反向、防作弊、自测、优化"等方法论要求——这些已内化在这里。

## 本仓库是什么

一个把**自然语言算法描述 → CUDA 前向+反向 kernel** 的 skill：以 PyTorch 参考实现为金标准通过正确性验证，
并在规范计时下超过 `torch.compile`。算法无关，每个算法是 `cases/<name>/` 下的一个 case。

## 当用户只给出一个算法定义时，你要自动做的事

用户的输入通常形如（等价于 `cases/<name>/description.md` 的内容）：
> "算法：LayerNorm。前向：每行减均值除标准差(含eps)再乘gamma加beta。输入 X[B,D]、gamma[D]、beta[D]，fp32。对 X/gamma/beta 求梯度。"

收到这类输入后，**不要等用户给更多指令**，直接按以下流程执行：

> **⚠️ 硬规则（先于步骤 0.5 判定，实测 codex 反复栽）——"点了算子名"不等于"规格已定、可跳确认"**：
> 很多算子有**多种标准变体**，点名不消除歧义：grid_sample（边界 zeros/border/reflection、align_corners、求梯度对象 input 还是 input+grid、bilinear/nearest）、segment_softmax/scatter（段有序否、values 是 [N] 标量还是 [N,D]、求梯度对象）、attention（causal/多头/scale）、conv（padding/stride/dilation/groups）、pool（kernel/stride/ceil_mode）等。
> **判定：无论用户是否点了明确算子名、是否给了部分公式，只要该算子存在这类"多解变体选项"（尤其求梯度对象/边界/维度/段序这类直接改变 reference 和反向的），就必须先走确认——列出你采取的每个变体默认 + 备选，呈请确认，未确认不得建 case。** 别因"我认得这算子、它有 PyTorch 默认语义"就跳过——PyTorch 默认≠用户要的默认（实测 codex 见"相当于 grid_sample"直接把"只对 input"擅自扩成"input+grid 都求"；见 grid_sample §31）。只有当**所有变体选项都已由用户明确给定**（前向公式/shape/grad_inputs/边界对齐等全明确）时才跳过确认。

**步骤 0.5（当用户只给真正的自然语言、不含数学公式/精确 shape 时）**：先**推导数学规格**呈请确认，再动工。
产出两部分给用户：① **结构化数学规格**——前向数学公式/伪代码 + 各输入名字/形状/dtype + 输出 + `grad_inputs` + 标量参数 + **⚠️语义澄清点**（凡自然语言有多种合理解释处，显式列出你采取的解释+备选，如"‘归一化’按 LayerNorm 理解，若要 L2 请指出"）；② 按规范写的 **PyTorch reference.py 代码**。**然后停下等用户确认或修正——用户未确认数学前，不得建 `cases/<name>/`、不得写 kernel**。这是唯一的人类确认闸门（只在数学层一次）；确认后按下面 1→5 **全自动跑完**不再中途停，确认过的数学规格即 `description.md`。
（若用户已给明确前向公式+shape/dtype **且无多解变体待定**，数学已定，**跳过步骤 0.5**直接下面第 1 步。）

1. **先读 `skill/SKILL.md`**（方法论主体，尤其"Case 协议""CUDA Kernel 实现技巧""防作弊红线""达标判据"）
   和 `skill/DESIGN.md`、`framework/case.py`。`framework/` 对你**只读**。
2. **命名并建 case**：据算法起一个简短 case 名（如 `layernorm`、`rmsnorm`），建 `cases/<name>/`，
   把用户的算法定义写成 `cases/<name>/description.md`（自然语言 + 前向数学 + shape/dtype + 对哪些输入求梯度）。
   **名字拿不准或可能与已有 case 重名时，先问用户一句**再继续。
3. **写实现**（严格按 SKILL.md "Case 协议"的 7 字段和骨架）：
   - `reference.py`：PyTorch 金标准，用基础算子表达前向（**禁止落回 `F.*`/SDPA 等高层算子**），autograd 提供反向。
     **必须向量化，禁止 Python 沿任何张量维度的 `for` 循环（包括时序/序列维度，不只是"逐元素/逐列"）**——描述里"单遍/在线扫描/逐列累加/沿时序递推"是数学语义不是实现方式，
     用广播+整体规约+`torch.cumsum`/`torch.pow(a,arange(T))` 等价表达。原因：bench 会对 reference 做 `torch.compile` 作 baseline，
     Python 循环展开成 O(N) 巨型图会**卡死 bench**（online_softmax 曾逐列 C=1024→编译 44s 挂；linear_ssm 曾 `for t in range(1024)` → 死锁 fork 进程炸弹），且逐元素 eager baseline 畸形慢造成**"弱 baseline 假象"**（加速比虚高不诚实）。**scan/递推类算法**（如 `h_t=a*h_{t-1}+b*x_t`）**禁止用 `torch.tril(W)+einsum` 的 O(T²) 密集矩阵变体**（伪向量化，把 O(T) 算法恶化到 O(T²)），要用 `torch.cumsum` 类 O(T) 前缀原语（如 `h_t=b·a^t·cumsum(x/a^t)`）。**reference 还禁止规模/条件专属分支**（`if x.numel()>=阈值: 快路径 else: 慢路径`）——金标准须始终用同一份最干净向量化，否则 bench 规模命中慢分支会造弱 baseline 假象（welford 曾因 `numel<64M 走 cumsum 慢分支` 刷出前2.71×，走干净分支真实仅0.99×）。reference 里出现 `for` 遍历任何张量维度、`T×T` 中间矩阵、或按 `numel/shape` 切实现分支，几乎一定写错。
     **例外：真变系数递推**（系数**输入依赖**、每步随输入变化，如门控 SSM/Mamba 类 `h_t=z_t·h_{t-1}+(1-z_t)·x_t`、`z_t` 依赖 `x_t`）**可能无数值稳定的 O(N) 向量化**——写成 `cumprod`/`cumsum` 的 O(N) 形式会因系数连乘下溢/上溢 NaN 或反向图畸形，`for t` 又编译爆炸。此时 **O(T²) 下三角权重矩阵是合法诚实 reference**（该算法数值稳定的最优可行 baseline，非"偷懒用 T² 代替本可 O(N)"）。区分：固定系数递推（linear_ssm）有稳定 O(N) 前缀形式→禁 T²；变系数递推若无稳定 O(N)→T² 合法。自主判断：先试 O(N)，verify 发现 NaN/溢出/反向异常慢再退 O(T²) 下三角。**另禁止用 `make_inputs` 挑异常输入分布迁就脆弱 reference**（如让某 sigmoid/exp 激活因偏置恒接近饱和值使 cumprod 不下溢——属把数值缺陷藏进输入，评测作弊；应用自然分布，若此时 reference 崩则改 reference 写法）。
   - `config.py`：shape/参数；**短核 case 让规模支持 env 覆盖**（如 `B = int(os.environ.get("LN_B","4096"))`）。
   - `__init__.py`：组装 `CASE`（7 字段）。
   - `kernels/*.cu`：前向 + 反向 kernel。**反向公式用户不会给——按 SKILL.md 技巧库自主推导**（autograd 对拍校验）。
   - `op.py`：`torch.autograd.Function` 封装为 `candidate(inputs, params)`。
   - **config.py 让规模支持 env 覆盖**（如 `B=int(os.environ.get("RMS_B","4096"))`）：这是**必做**的一小步——
     harness 的自动放大兜底靠它。**短核 case 可选加 `cases/<name>/bench.env`**（内容一行 `SIZE_ENV="<你的规模env>=32768"`）：
     若默认规模下核很短（前/反向 <0.15ms，如归一化/逐元素类默认 B=4096 时前向仅 0.06ms），计时固定开销会把加速比抬虚高。
     **即便你不建 bench.env，`run_on_a100.sh` 也会自动探测短核并用 config 的规模 env 放大重测**（harness 兜底）——
     但建了 bench.env 能省一次探测、更明确。config 支持 env 是前提，别把规模写死。
4. **自测（自动，无需用户提）**：跑 `bash skill/scripts/run_on_a100.sh <name> --gpu 7 --strict`
   （首次加 `--sync-cli`）。它在远程 GPU 跑 verify+bench，末行给 `VERDICT=`。按 `skill/AUTONOMOUS_LOOP.md` 的
   VERDICT 决策：`PASS`→交付；`VERIFY_FAIL`→修正确性（不看 bench）；`BENCH_FAIL`→按 `skill/loop.md` 优化未达标侧 kernel；
   `CV_INVALID`→原样重跑。
   > **警惕短核假象**：若 bench 显示 baseline 前向/反向 <0.15ms 却给出高加速比（如 1.2×+），**别信、别停**——
   > 那是固定开销虚高。回去建/改 bench.env 放大规模重测，看真实加速比再判达标。
5. **优化到达标**：未达标则迭代（只改 `cases/<name>/`），直到 `VERDICT=PASS`。
   **擦线（1.05–1.10×）须连跑 3 次全 PASS 才算达标**（见 SKILL.md 达标判据）。

## 防作弊红线（不可违反，详见 SKILL.md）

1. 待测路径禁止落回 `F.scaled_dot_product_attention` / `torch.nn.functional` / `torch.matmul` / `torch.sparse.mm` 等 torch 高层算子。**但允许 CUDA 官方底层库**（cuBLAS/CUB/cuSPARSE——candidate 在自定义 `.cu` 里调它们+手写融合 kernel 合规，非回到 torch 高层）。**通用张量原语**（`torch.topk`/`sort`/`cumsum`/`scatter_add`/`index_select`）**reference 里允许**（基础操作非层算子），但 candidate 仍须手写 `.cu` 不得直接调糊弄；**神经网络层算子**（`F.*`/`nn.*`：max_pool/layer_norm/conv/sdpa/embedding）reference 也禁。**⚠️ 库调用"辅助原语 vs 直调目标算子"细分（Cholesky 实测,aider 钻空子）**：cuBLAS/cuSOLVER 只能作**辅助原语**（GEMM/TRSM/AXPY/scan 等积木,candidate 拼算法+手写调度）;**禁直调"与 case 目标算子语义等价的库成品"**（Cholesky 直调 `cusolverDnSpotrf`、解线性系统直调 `getrf/gesv`、QR `geqrf`、FFT cuFFT 等）——那 candidate 就是 baseline 同款厂商算法,失去"手写 kernel 跑赢"意义（等于抄 baseline）。判据:库调用是**积木**(还需拼)还是**成品**(直接是本题答案)——成品级直调禁。厂商库墙形态(baseline 就是 cuSOLVER/cuFFT)正解是手写尽力+诚实报边界,非直调同库假装赢。
2. `framework/` 只读——禁止修改/绕过验证器、计时器、协议。
3. 禁止降精度换速度（保持 fp32、不用 fast-math）。
4. 交付 `.cu` 须能独立编译、无 torch 高层运行时依赖。
5. **评测路径必须=真实路径**：`op.py` 的 `candidate` 不得针对评测的测量方式（bench 用 `no_grad()`+`detach()` 计前向）
   走真实使用不会走的快路径（如无梯度就绕过 autograd、跳过反向所需中间量存储）。前向提速只能来自 kernel 本身。

## GPU 自测环境（已就绪，见 `skill/AUTONOMOUS_LOOP.md`）

`run_on_a100.sh` 经双跳 SSH 直传远程 A100 跑评测。若你在 WSL，需先把密钥拷进 WSL：
`cp /mnt/c/Users/<user>/.ssh/nl2cuda_gpu ~/.ssh/ && chmod 600 ~/.ssh/nl2cuda_gpu`（Windows 侧免拷）。

## 一句话总结

**用户给算法定义 → 你读 SKILL.md → 自建 case → 写 reference/kernel/op（自主推导反向）→ run_on_a100.sh 自测 →
按 VERDICT 迭代到稳定 PASS → 守全部防作弊红线。全程不必等用户逐步指令。**
