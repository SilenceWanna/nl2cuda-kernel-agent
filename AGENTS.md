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

1. **先读 `skill/SKILL.md`**（方法论主体，尤其"Case 协议""CUDA Kernel 实现技巧""防作弊红线""达标判据"）
   和 `skill/DESIGN.md`、`framework/case.py`。`framework/` 对你**只读**。
2. **命名并建 case**：据算法起一个简短 case 名（如 `layernorm`、`rmsnorm`），建 `cases/<name>/`，
   把用户的算法定义写成 `cases/<name>/description.md`（自然语言 + 前向数学 + shape/dtype + 对哪些输入求梯度）。
   **名字拿不准或可能与已有 case 重名时，先问用户一句**再继续。
3. **写实现**（严格按 SKILL.md "Case 协议"的 7 字段和骨架）：
   - `reference.py`：PyTorch 金标准，用基础算子表达前向（**禁止落回 `F.*`/SDPA 等高层算子**），autograd 提供反向。
   - `config.py`：shape/参数；**短核 case 让规模支持 env 覆盖**（如 `B = int(os.environ.get("LN_B","4096"))`）。
   - `__init__.py`：组装 `CASE`（7 字段）。
   - `kernels/*.cu`：前向 + 反向 kernel。**反向公式用户不会给——按 SKILL.md 技巧库自主推导**（autograd 对拍校验）。
   - `op.py`：`torch.autograd.Function` 封装为 `candidate(inputs, params)`。
4. **自测（自动，无需用户提）**：跑 `bash skill/scripts/run_on_a100.sh <name> --gpu 7 --strict`
   （首次加 `--sync-cli`）。它在远程 GPU 跑 verify+bench，末行给 `VERDICT=`。按 `skill/AUTONOMOUS_LOOP.md` 的
   VERDICT 决策：`PASS`→交付；`VERIFY_FAIL`→修正确性（不看 bench）；`BENCH_FAIL`→按 `skill/loop.md` 优化未达标侧 kernel；
   `CV_INVALID`→原样重跑。
5. **优化到达标**：未达标则迭代（只改 `cases/<name>/`），直到 `VERDICT=PASS`。
   **擦线（1.05–1.10×）须连跑 3 次全 PASS 才算达标**（见 SKILL.md 达标判据）。

## 防作弊红线（不可违反，详见 SKILL.md）

1. 待测路径禁止落回 `F.scaled_dot_product_attention` / `torch.nn.functional` 等高层算子。
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
