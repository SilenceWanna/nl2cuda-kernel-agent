# Agent × Case 测试矩阵（prompt 与流程）

用于在多个宿主 agent 上、用**精简 description + skill 技巧库**驱动其重新生成各 case 的
CUDA 前反向 kernel，验证 skill 的宿主无关性与"知识内置"（agent 靠 skill 自主推导，不靠用户喂公式）。

> **维护约定**：新增 case 时，在 §3 加一段该 case 的任务 prompt；新增 agent 时，在 §2 加其启动方式。
> 矩阵勾选表（§4）同步更新。

## 1. 通用准备：干净房间

每个 (agent × case) 测试都在"干净房间"里做——只有精简 description，无任何实现，避免 agent 抄现成：

```bash
# 新 clone 一份，删掉所有 case 实现只留 description.md，并预建空文件（解某些 agent 的写盘卡顿）
git clone https://github.com/SilenceWanna/nl2cuda-kernel-agent.git <workdir>
cd <workdir>
for c in rbf layernorm softmax_ce; do
  rm -f cases/$c/reference.py cases/$c/config.py cases/$c/__init__.py cases/$c/op.py cases/$c/kernels/*.cu
  for f in reference.py config.py __init__.py op.py; do : > cases/$c/$f; done
done
rm -rf cases/rbf/delivery
# 各 case 的空 kernel 文件按其命名预建，如：
: > cases/rbf/kernels/rbf_forward.cu; : > cases/rbf/kernels/rbf_backward.cu
: > cases/layernorm/kernels/layernorm_forward.cu; : > cases/layernorm/kernels/layernorm_backward.cu
: > cases/softmax_ce/kernels/softmax_ce_forward.cu; : > cases/softmax_ce/kernels/softmax_ce_backward.cu
```

每个 agent 用**独立 workdir**（避免并行抢目录），产物推到独立分支 `test/kt-<case>-<agent>`。

## 2. 各 Agent 启动方式（均连京东内网 OpenAI 兼容代理 DongCC `127.0.0.1:8787`）

> 京东代理踩坑：`GPT-5.5-joybuilder` 不接受 `temperature`（否则 400）；`DeepSeek-V3` 接受 temperature 但拒 content 数组。按 agent 选模型/配置规避。

### aider（本地 Windows / 可用）
```powershell
cd <workdir>
$env:PATH = "C:\Users\<user>\.local\bin;$env:PATH"
$env:OPENAI_API_BASE = "http://127.0.0.1:8787/v1"
$env:OPENAI_API_KEY  = "<代理 key>"
aider --model openai/GPT-5.5-joybuilder --model-settings-file .aider.model.settings.yml --yes-always --no-show-model-warnings
```
`.aider.model.settings.yml` 内容（禁 temperature）：
```yaml
- name: openai/GPT-5.5-joybuilder
  use_temperature: false
  streaming: true
```
写盘卡顿：新建文件时 aider 可能陷入 file-not-found 反射循环——**预建空文件**（§1）可解。

### codex（桌面/CLI）
配置 `~/.codex/config.toml` 的 provider 指向 `http://127.0.0.1:8787/v1`、model `GPT-5.5-joybuilder`。
codex 走 responses API，接口一次对（会主动读 framework/case.py）。工作目录设为 <workdir>。

### gptme（WSL / 本地无法，Windows 缺 termios）
```bash
cd ~/<workdir-in-wsl>
export PATH="$HOME/.local/bin:$PATH"
export OPENAI_BASE_URL="http://127.0.0.1:8787/v1"   # WSL 可直连 Windows 的 127.0.0.1:8787
export OPENAI_API_KEY="<代理 key>"
gptme --model openai/GPT-5.5-joybuilder
```
temperature 坑：gptme 靠 `model_meta.supports_reasoning` 跳过 temperature；GPT-5.5 是 unknown model
走 fallback（supports_reasoning=False）会发 temperature → 400。已 patch 其 `models.py` fallback
为 `supports_reasoning=True`。

## 3. 各 Case 任务 Prompt（精简 description，让 agent 自主推导反向）

所有 case 通用前缀（喂给 agent）：
```
本仓库是"自然语言→CUDA前反向kernel"的 skill。请先读 skill/SKILL.md（尤其"Case 协议"和
"CUDA Kernel 实现技巧"两章）、skill/DESIGN.md、framework/case.py。framework/ 只读禁改。
据 cases/<CASE>/description.md 实现该算法的 CUDA 前向+反向 kernel，创建 cases/<CASE>/ 下的
reference.py、config.py、__init__.py、op.py、kernels/*.cu（文件已存在为空）。
注意：description 只给前向定义和"对哪些输入求梯度"，反向公式没给——请按 SKILL.md 技巧库
自主推导（或用 autograd 参考值对拍）。约束：fp32、不用 fast-math、不落回 F.* 高层算子。
```

### RBF（成对距离/高斯核）
`<CASE>=rbf`。输入 X:[N,D],Y:[M,D] → K:[N,M]=exp(-γ‖x_i-y_j‖²)；对 X、Y 求梯度。
反向要点（agent 自推）：`S=-γ·G·K`，`dX[i]=Σ_j S·2(x_i-y_j)`、`dY[j]=Σ_i S·2(y_j-x_i)`；可缓存 K。

### LayerNorm（层归一化）
`<CASE>=layernorm`。输入 X:[B,D],gamma/beta:[D] → 同形 Y；对 X、gamma、beta 求梯度。
反向要点（agent 自推）：dgamma/dbeta 是列规约（沿 B），dX 有耦合项——见技巧库"列规约优化"。

### Softmax-CE（softmax 交叉熵）
`<CASE>=softmax_ce`。输入 logits:[B,C]、labels:[B](int64) → 标量 loss；只对 logits 求梯度。
反向要点（agent 自推）：`dlogits=(softmax-onehot)/B`；前向 logsumexp 先减 max（数值稳定）。

## 4. 验证（每个产物推分支后，在 A100 上跑）

```bash
# A100（经跳板机双跳）：拉分支 → verify（正确性）→ bench（性能）
cd ~/nl2cuda-kernel-agent && git fetch origin && git checkout test/kt-<case>-<agent> && git pull --ff-only
export PATH=~/miniconda3/bin:/usr/local/cuda/bin:$PATH
CUDA_VISIBLE_DEVICES=<空闲卡> CUDA_ARCHS=80 python skill/scripts/verify_case.py --case <case>
CUDA_VISIBLE_DEVICES=<空闲卡> CUDA_ARCHS=80 python skill/scripts/bench_case.py  --case <case>
```
判据：verify 前反向全 PASS（allclose）；bench 前反向各 ≥1.05×（CV>5% 属共享卡噪声，重测）。

## 5. 矩阵勾选表（3 case × 3 agent；✅正确性PASS ⚡达标 ⬜未测）

| case \ agent | aider | codex | gptme |
|--------------|-------|-------|-------|
| RBF          | ✅（精简版重生 PASS，前~2e-7/反~6e-7，用了缓存K复用） | ⬜ | ⬜ |
| LayerNorm    | ✅（5.3 已验证 PASS，自主推导dX耦合项） | ⬜ | ⬜ |
| Softmax-CE   | ⬜（待 aider 生成） | ✅⚡（前1.97/反1.80，早期非精简版） | ⬜ |

> 注：codex/gptme 此前的 Softmax-CE 达标是在 description 精简**之前**测的；本轮矩阵用精简版重测以对齐。
> 分支命名：`test/kt-<case>-<agent>`（如 `test/kt-rbf-aider`）。
