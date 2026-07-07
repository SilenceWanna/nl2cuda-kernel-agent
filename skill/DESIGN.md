# Skill 设计文档：自然语言 → CUDA 前反向 Kernel 生成

> 本文件是 skill 的架构与契约说明（面向开发者/维护者）。方法论主体见 [SKILL.md](SKILL.md)，迭代循环见 [loop.md](loop.md)。

## 1. 目标与定位

交付一个 **agent skill**，让宿主 agent（先 Claude Code，后续可移植 Codex/开源 agent）对**任意算法结构**自动完成闭环：

```
自然语言算法描述(+shape/dtype) → PyTorch 参考实现 → CUDA 前向+反向 kernel → 正确性验证 → 计时优化 → 交付 .cu
```

**通用性是最终目标**；RBF 高斯核矩阵只是**第一个验收 case**，后续会加更多算法。因此架构严格分两层，framework 层不得出现任何算法特定内容。

## 2. 两层架构

```
framework/   通用框架（算法无关）——绝不出现某个具体算法的名字/形状
  protocol.py   全局计时/容差协议（warmup/iters/repeats/cv/target/atol/rtol/seeds）
  loader.py     CUDA 扩展即时编译加载（sm_75，默认不开 fast-math）
  verify.py     verify_forward/backward(case, impl) —— 从 case 取输入/参数/梯度键
  bench.py      compare(case, candidate) —— upstream 梯度按输出形状 randn_like
  smoke.cu / smoke_test.py   编译链路冒烟

cases/<name>/   算法 case（每算法一份，完全可替换）
  __init__.py   暴露 CASE: Case 实例
  reference.py  PyTorch 金标准（前向；autograd 提供反向）
  config.py     该算法的 shape / 标量参数 / 形状相关设置
  description.md 自然语言算法描述（skill 的输入样例）
  kernels/      该算法的 CUDA kernel（agent 生成物；RBF 附阶段1参考解作示例）
  op.py         用 autograd.Function 把 kernel 前反向包装成候选实现

skill/
  SKILL.md      方法论主体（算法无关，宿主 agent 读它执行）
  DESIGN.md     本文件
  loop.md       迭代循环与终止条件
  scripts/      通用 CLI：verify_case.py / bench_case.py（--case <name>）
```

## 3. Case 协议（framework 与 case 的唯一接口）

一个 case 是实现了以下属性/方法的对象（`framework/verify.py`、`bench.py` 只依赖此协议）：

```python
class Case:
    name: str                      # "rbf"
    description: str               # 自然语言描述（喂给 agent）
    params: dict                   # 算法标量参数，如 {"gamma": 1/64}
    grad_inputs: list[str]         # 需求梯度的输入名，如 ["X", "Y"]
    dtype: str                     # 验收精度，如 "float32"

    def make_inputs(seed, dtype, device, requires_grad) -> dict[str, Tensor]:
        # 返回命名输入张量，如 {"X":..., "Y":...}；键名即梯度归属

    def reference_forward(inputs: dict, params: dict) -> Tensor:
        # 金标准前向（广播/朴素写法，autograd 可反向）；返回输出张量
```

**候选实现（agent 生成 kernel 的封装）的统一接口**：

```python
def candidate(inputs: dict, params: dict) -> Tensor
    # 前向；对 grad_inputs 中的输入，output.backward(G) 后 inputs[name].grad 被填充
```

framework 用 `inputs`/`grad_inputs` 泛化，不假设输入个数或输出形状——这是支持任意算法的关键。

## 4. Agent 工作流（SKILL.md 展开）

1. **读描述**：解析 NL 算法描述 + shape/dtype，明确输入/输出/参数/待求梯度。
2. **写 PyTorch 参考**：用基础算子（广播/matmul/逐元素）表达，**禁止落回 SDPA/`F.*` 高层融合算子**；autograd 提供反向。作为正确性金标准。
3. **写 CUDA 前向 kernel** → 用 `verify_case.py` 对拍参考前向（≥5 种子 allclose）。
4. **写 CUDA 反向 kernel** → autograd.Function 包装 → 验证反向各梯度 allclose。
5. **计时** `bench_case.py` 对比 `torch.compile`：前反向是否各 ≥1.05×。
6. **未达标 → 进入优化循环**（见 loop.md）：读 profile → tiling/float4/warp/融合 → 重验证重计时。
7. **交付**：可独立编译的 `.cu`（含 host 绑定），无 torch 高层运行时依赖。

## 5. 工具契约（agent 只读，进程隔离）

- `python skill/scripts/verify_case.py --case <name> [--impl <module:fn>]`
  - 输出每种子前/反向 allclose 结果 + max_abs_err，末尾 `PASS`/`FAIL`。
- `python skill/scripts/bench_case.py --case <name> [--impl <module:fn>]`
  - 输出候选 vs torch.compile 前反向几何均值/CV/加速比，末尾达标判定。
- **agent 禁止修改 framework/（验证器、计时器、协议）**——这是防作弊的只读评测基座。agent 只在 `cases/<name>/kernels/` 和 `op.py` 内写 kernel 与封装。

## 6. 失败反馈回喂格式

工具 stdout 设计为 agent 可直接读取的结构化文本：
- 编译失败 → nvcc 报错原文（agent 据此改 kernel 语法）。
- 正确性失败 → 每种子 max_abs_err（agent 判断是数值/索引/边界错）。
- 计时未达标 → 前反向各自加速比 + baseline 绝对耗时（agent 判断优化方向）。
- CV 超阈值 → 提示结果作废需重测（环境噪声，非 kernel 问题）。

## 7. 防作弊红线（不可违反）

- 待测路径禁止落回 `F.scaled_dot_product_attention` / `torch.nn.functional` 等高层算子。
- 禁止修改/绕过 `framework/` 下的验证与计时脚本。
- 禁止降精度换速度（除非算法描述本身指定低精度）。
- 交付 `.cu` 须能独立编译、无 torch 高层运行时依赖。

## 8. 计时协议（framework/protocol.py，全局统一）

warmup≥10；正式≥100 取几何均值；CUDA events 计时，每次前后 synchronize；3 次重跑算 CV，CV>5% 作废重测；**前向、反向分别计时各自达标**；baseline = `torch.compile`(默认 mode)；达标阈值 1.05×。跨 case 统一，保证公平对比。
