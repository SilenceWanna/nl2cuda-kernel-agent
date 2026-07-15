# CLAUDE.md — Claude Code 工作约定（自动注入上下文）

> 本文件内容与 `AGENTS.md` 一致（单一事实来源的镜像；有出入以 `AGENTS.md` 为准）。
> **目的**：用户**只需输入算法定义**，你就按本约定 + `skill/SKILL.md` 自动走完全流程，无需用户逐步指令。

## 本仓库

自然语言算法描述 → CUDA 前向+反向 kernel 的 skill：PyTorch 参考为金标准过正确性验证，规范计时超 `torch.compile`。
算法无关，每算法是 `cases/<name>/` 下一个 case。

## 当用户只给出一个算法定义时（如"算法：LayerNorm。前向…。输入…fp32。对 X/gamma/beta 求梯度。"）

**不要等用户给更多指令**，直接执行：

1. 先读 `skill/SKILL.md`（尤其"Case 协议""CUDA Kernel 实现技巧""防作弊红线""达标判据"）、`skill/DESIGN.md`、`framework/case.py`。`framework/` 只读。
2. 据算法起简短 case 名，建 `cases/<name>/`，把定义写成 `description.md`。名字拿不准/可能重名先问一句。
3. 严格按 Case 协议 7 字段写 `reference.py`（基础算子、禁 `F.*` 高层算子、**必须向量化禁 Python 逐元素/逐列 `for` 循环**——描述里"单遍/在线扫描"是数学语义，用广播+规约表达；否则 bench 对其 `torch.compile` 会卡死+弱 baseline 假象）、`config.py`（短核让规模支持 env 覆盖）、`__init__.py`、`kernels/*.cu`（**反向按技巧库自主推导，autograd 对拍**）、`op.py`（`candidate`）。
4. **自测（自动，无需用户提）**：跑 `bash skill/scripts/run_on_a100.sh <name> --gpu 7 --strict`（首次加 `--sync-cli`），据 `VERDICT=`/日志按 `skill/AUTONOMOUS_LOOP.md` 决策。
5. 未达标按 `skill/loop.md` 优化（只改 `cases/<name>/`）到 `VERDICT=PASS`；擦线（1.05–1.10×）须连跑 3 次全 PASS。

## 防作弊红线（不可违反，详见 SKILL.md）

1. 禁落回 `F.scaled_dot_product_attention`/`torch.nn.functional` 等高层算子。
2. `framework/` 只读（禁改验证器/计时器/协议）。
3. 不降精度换速度（fp32、无 fast-math）。
4. 交付 `.cu` 独立编译、无 torch 高层运行时依赖。
5. **评测路径=真实路径**：`op.py` 的 `candidate` 不得针对 bench 的 `no_grad`+`detach` 计时走真实使用不会走的快路径（如无梯度绕过 autograd、跳过反向所需中间量存储）；前向提速只能来自 kernel 本身。

## GPU 自测环境

`run_on_a100.sh` 经双跳 SSH 直传远程 A100 跑评测（Windows 侧密钥已就绪；WSL 需先拷密钥）。详见 `AGENTS.md` 与 `skill/AUTONOMOUS_LOOP.md`。
