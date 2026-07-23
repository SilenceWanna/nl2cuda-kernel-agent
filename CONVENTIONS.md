# CONVENTIONS.md — aider 工作约定（aider 自动加载）

> 本文件内容与 `AGENTS.md` 一致，是单一事实来源的镜像；如有出入以 `AGENTS.md` 为准。
> **目的**：用户在 aider 界面**只需输入算法定义**，你就按本约定 + `skill/SKILL.md` 自动走完全流程。

## 当用户只给出一个算法定义时（如"算法：LayerNorm。前向…。输入…fp32。对 X/gamma/beta 求梯度。"）

**不要等用户给更多指令**，直接执行：

0.5. **当用户只给真正的自然语言、不含数学公式/精确 shape 时**（如"把每行归一化再缩放""做个带因果 mask 的注意力"）：先**推导数学规格**呈请确认再动工——给用户 ① 结构化数学规格（前向公式/伪代码 + 输入输出 shape/dtype + `grad_inputs` + 标量参数 + **⚠️语义澄清点**：凡有多种合理解释处显式列出采取的解释+备选，如"‘归一化’按 LayerNorm 理解，要 L2 请指出"）+ ② PyTorch reference.py 代码，**然后停下等确认；未确认不得建 case/写 kernel**。这是唯一人类确认闸门（数学层一次），确认后按下面 1→5 全自动跑完不再中途停，确认过的规格即 `description.md`。（用户已给明确公式+shape 则跳过本步。）

> **⚠️ 硬规则（先于 0.5 判定，实测 codex 反复栽）——"点了算子名"≠"规格已定、可跳确认"**：很多算子有多种标准变体、点名不消歧义：grid_sample（边界 zeros/border/reflection、align_corners、求梯度对象 input 还是 input+grid）、segment_softmax/scatter（段有序否、values 是 [N] 还是 [N,D]、求梯度对象）、attention（causal/多头/scale）、conv（padding/stride/groups）、pool 等。**无论是否点了明确算子名、是否给了部分公式，只要该算子有这类多解变体（尤其求梯度对象/边界/维度/段序这类直接改 reference 和反向的），就必须先按 0.5 列变体默认+备选呈请确认，未确认不得建 case。** 别因"认得这算子、有 PyTorch 默认语义"就跳——PyTorch 默认≠用户要的默认（实测 codex 见"相当于 grid_sample"擅自把"只对 input"扩成"input+grid 都求"）。仅当所有变体选项都已明确给定才跳过。

1. 先读 `skill/SKILL.md`（尤其"Case 协议""CUDA Kernel 实现技巧""防作弊红线""达标判据"）、`skill/DESIGN.md`、`framework/case.py`。`framework/` 只读。
2. 据算法起简短 case 名，建 `cases/<name>/`，把定义写成 `description.md`。名字拿不准/可能重名先问一句。
3. 严格按 Case 协议 7 字段写 `reference.py`（基础算子、禁 `F.*` 高层算子、**必须向量化禁 Python 沿任何张量维度的 `for` 循环（含时序/序列 for）**——描述里"单遍/在线扫描/沿时序递推"是数学语义，用广播+规约+`torch.cumsum` 表达；scan/递推类禁 `torch.tril+einsum` 的 O(T²) 密集矩阵伪向量化，要用 O(T) 前缀原语；**且禁规模/条件专属分支（`if numel>=阈值:快 else:慢`），须始终单一最干净向量化**；**例外：真变系数递推（系数输入依赖，如门控 SSM `z_t=sigmoid(w·x_t+b)`）可能无稳定 O(N) 向量化（cumprod 下溢 NaN/反向畸形、for 编译爆炸），此时 O(T²) 下三角合法诚实；自主判断先试 O(N) 崩了再退 T²。且禁用 make_inputs 挑异常分布迁就脆弱 reference**；否则 bench 对其 `torch.compile` 会卡死或弱 baseline 假象）、`config.py`（短核让规模支持 env 覆盖）、`__init__.py`、`kernels/*.cu`（**反向自主推导**）、`op.py`（`candidate`）。
4. **自测（自动）**：已配 `--auto-test`，你每次改完会自动跑 `run_on_a100.sh <name> --strict`；据其 `VERDICT=`/日志按 `skill/AUTONOMOUS_LOOP.md` 决策。也可手动 `/run bash skill/scripts/run_on_a100.sh <name> --gpu 7 --strict`。
5. 未达标按 `skill/loop.md` 优化（只改 `cases/<name>/`）到 `VERDICT=PASS`；擦线（1.05–1.10×）须连跑 3 次全 PASS。

## 防作弊红线（不可违反，详见 SKILL.md）

1. 禁落回 `F.scaled_dot_product_attention`/`torch.nn.functional` 等高层算子。
2. `framework/` 只读。
3. 不降精度换速度（fp32、无 fast-math）。
4. 交付 `.cu` 独立编译、无 torch 高层运行时依赖。
5. **评测路径=真实路径**：`op.py` 的 `candidate` 不得针对 bench 的 `no_grad`+`detach` 计时走真实使用不会走的快路径（如无梯度绕过 autograd、跳过反向所需中间量存储）；前向提速只能来自 kernel。

> 完整流程与 GPU 自测环境说明见 `AGENTS.md` 与 `skill/AUTONOMOUS_LOOP.md`。
