# 在其他 Agent 上使用本 Skill（宿主无关性测试）

本 skill（`SKILL.md` 方法论 + `scripts/` 工具 + `framework/` 评测基座 + `cases/` 算法实例）
设计为**宿主无关**：任何能读文档、执行 shell、访问 GPU 的 agent 都能驱动它完成
"自然语言算法描述 → PyTorch 参考 → CUDA 前反向 kernel → 验证 → 计时优化 → 交付 .cu"。

已在 **Claude Code** 上完整验证（RBF 达标、LayerNorm 通用性验证）。本文档说明如何在
**Codex / Cursor / Cline / aider / 其他开源 agent** 上复现。

## 前提：三样东西

1. **仓库**：`git clone https://github.com/SilenceWanna/nl2cuda-kernel-agent.git`（public，无需认证）。
2. **带 GPU 的 CUDA + PyTorch 环境**（**唯一难点**）：
   - 需要 NVIDIA GPU、nvcc、PyTorch(cuda)、ninja。
   - 若 agent 与你在**同一台机器**运行，可复用你现有的 GPU 访问方式（如 SSH 到远程 GPU 机——环境已搭好，只需 `git pull`）。GPU 的具体连接方式请你私下提供给 agent（不写进公开仓库）。
   - 若 agent 在自己的沙箱运行，需自备 GPU 环境：`pip install torch --index-url https://download.pytorch.org/whl/cu118`（老系统 GCC<9 用 `torch==2.0.1`）、`pip install ninja`、确保 nvcc 在 PATH、`CUDA_VISIBLE_DEVICES` 指向空闲卡。
3. **一个算法的自然语言描述**（测试输入）。

## 给 agent 的上手提示（复制粘贴，替换尖括号内容）

```
你将使用一个名为 nl2cuda-kernel 的 skill，为我把一个算法结构实现为自定义 CUDA
前向+反向 kernel，并让它在规范计时下超过 torch.compile。

1. 先读 skill/SKILL.md（方法论主体）和 skill/DESIGN.md（架构），严格按其流程与
   防作弊红线执行。评测基座 framework/ 对你只读，禁止修改。
2. 算法描述：<把新算法的自然语言描述+shape/dtype 贴这里，或指向 cases/<name>/description.md>
3. 按 SKILL.md 步骤：在 cases/<name>/ 下写 PyTorch 参考(reference.py，禁止落回
   F.*/SDPA 等高层算子)、config.py、__init__.py 暴露 CASE；再在 kernels/ 写前向
   与反向 .cu，用 op.py 的 autograd.Function 封装为 candidate。
4. 用只读工具自检（GPU 环境下运行）：
     python skill/scripts/verify_case.py --case <name>     # 前反向 allclose 全 PASS
     python skill/scripts/bench_case.py  --case <name>     # 前反向各 ≥1.05× torch.compile
     python skill/scripts/profile_case.py --case <name>    # 找瓶颈（未达标时）
5. 未达标则按 skill/loop.md 迭代优化 kernel（只改 cases/<name>/，不动 framework/）。
6. 约束：fp32 全精度、不用 fast-math、不降精度换速度、不改评测脚本。
```

## 判断 skill 是否"在该 agent 上成功"

- ✅ agent 仅凭 SKILL.md 就能走完流程（不需要你手把手指导每一步）。
- ✅ 产出的 case 通过 `verify_case.py`（前反向 allclose 全 PASS）。
- ✅（进阶）`bench_case.py` 前反向达到 ≥1.05×。
- ✅ 全程未违反防作弊红线。

## 运行命令要点（GPU 环境）

评测工具是纯 python CLI，任何 agent 都能调：
```
python skill/scripts/verify_case.py --case <name>
python skill/scripts/bench_case.py  --case <name>
```
GPU 环境需正确设置 `CUDA_VISIBLE_DEVICES`（指向空闲卡）、nvcc 在 PATH、ninja 可用。
新增算法只需加 `cases/<name>/`，framework 与 CLI 无需改动。
