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

> **⚠️ 干净房间必须 commit 空实现（血泪教训）**：仅在工作区删实现文件、不 commit，是**假干净房间**——
> 因为 main/HEAD 里三个 case 一直带着达标实现，agent 跑 git 操作（或 clone 自 main）会还原出实现，
> 于是它"抄现成"而非重生（表现：产物与旧版 `git diff` 为 0，verify/bench 全过但无意义）。
> **正解**：在 workdir 建 cleanroom 分支，`git rm` 实现文件后**提交这个空状态**（`git commit`），
> 使 HEAD 里 reference/op/kernel 全为 0 字节。这样 agent 无论怎么操作都还原不出实现，只能靠精简 description+skill 自写。
> 验真伪：agent 产物与旧版 `git diff origin/main -- <kernel>.cu` 应有大量差异（0=又抄了）。

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
中文编码坑：gptme 处理中文 prompt 报 `'ascii' codec can't encode`——需 `export PYTHONIOENCODING=utf-8 PYTHONUTF8=1`
（即便 WSL LANG=C.UTF-8，Python stdio 仍可能按 ASCII）。目录名区分大小写，用小写 `cases/rbf`。
**措辞坑**：GPT-5.5 经 gptme 对 prompt 措辞敏感——"请先读...然后..."这类温和/分步措辞会让它只输出计划、不调工具（文件空转）。需**命令式**："现在立即用工具创建并写入这些文件，不要只描述计划、不要只输出代码块，每个文件实际写盘，写完逐一确认"。

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

> **A100↔GitHub 网络踩坑**：A100 到 github.com:443 时好时坏（常 `Connection timed out`/`Encountered end of file`），
> `git fetch` 可能悄悄失败却让 checkout 落回旧残留分支、verify 跑错代码（数字与目标 agent 对不上即为此）。
> 双跳 SSH 本身是通的，故可绕开 GitHub **直传**：本地 `tar -czf` 打包 case 文件 → `cat tar | ssh <双跳> "tar -xzf -"`
> 解压覆盖 → **清扩展缓存** `rm -rf ~/.cache/torch_extensions/*/<ext>` 再 verify（否则复用旧 .so）。这即阶段 6 的直传路线。

> **短核固定开销陷阱（LayerNorm 实测教训）**：case 默认规模若让核很短（如 LayerNorm 默认 B=4096 前向仅 0.06ms），
> baseline 的 kernel-launch 等固定开销会被摊进短核里造成**加速比虚高**（前向假象 1.2×），且 baseline CV 极易破 5%。
> 放大规模（`LN_B=32768` 拉到 0.23ms）摊薄固定开销后暴露真实性能（前向实为 0.97×）。
> **教训**：短核 case 计时前先放大规模让核跑到 ≥0.2ms 再判达标，否则数字不可信。阶段 6 agent 自主判达标时尤须内建此规则。

## 5. 矩阵勾选表（3 case × 3 agent；✅正确性PASS ⚡达标 ⬜未测）

| case \ agent | aider | codex | gptme |
|--------------|-------|-------|-------|
| RBF          | ✅（精简版重生 PASS，前~2e-7/反~6e-7，用了缓存K复用） | ✅⚡（精简版重生，前1.09×/反1.40×，tiling+coarsening+缓存K，CV<1%） | ✅⚡（重生首轮正确但反0.016×慢63倍；**阶段6自主闭环优化达标 前3.43×/反1.26×**——gptme 凭 VERDICT+日志自主诊断：移除atomicAdd竞争改独立累加、前向加D=64 shared tiling，已独立复验） |
| LayerNorm    | ✅（5.3 已验证 PASS，自主推导dX耦合项） | ⚠️→✅⚡（重生正确性全PASS+自主推导dX耦合项+二维分块列规约；前向短核0.97×未达标→**阶段6 codex 前向优化闭环自主2轮达标 前1.08×/反1.25×**：加D==1024 fast path(float4读+寄存器复用+warp shuffle规约)，已独立复验） | ⚠️→✅正确（重生首轮block_reduce广播bug致verify FAIL；**阶段6 aider 纠错闭环自主两轮修复**→verify全PASS。修后前向0.89×未达标(反1.22×)，属前向优化任务） |
| Softmax-CE   | ✅⚡（精简版重生，前1.23×/反2.55×，logsumexp减max+单文件.cu+缓存probs，前向前4种子误差0） | ✅⚡（精简版重生，前1.62×/反2.79×，logsumexp减max+template复用+缓存probs；早期非精简版曾1.97/1.80） | ✅⚡（真净房重生，前1.75×/反3.91×，logsumexp减max+block_reduce广播已修+缓存logsumexp反向重算softmax；反向为三agent最快） |

> 注：codex/gptme 此前的 Softmax-CE 达标是在 description 精简**之前**测的；本轮矩阵用精简版重测以对齐。
> 分支命名：`test/kt-<case>-<agent>`（如 `test/kt-rbf-aider`）。

## 6. 3×3 矩阵结论（精简 description 重生）

**9/9 全部生成，正确性 7/9 PASS，性能达标 6/9。** 三 agent 均能仅凭精简 description + SKILL.md 技巧库
**自主推导反向**（RBF 的 dX/dY、LayerNorm 最难的 dX 耦合项、Softmax-CE 的 dlogits），且守防作弊红线
（fp32/无 fast-math/无 F.*/framework 未改）。**知识转移成立**——反向公式已从"用户输入"移到"skill 方法论"。

agent 间差异**只在实现成熟度**（skill 无关）：
- **codex**：3/3 正确，2 达标（RBF 前1.09/反1.40、Softmax-CE 前1.62/反2.79）；LayerNorm 前向短核陷阱下 0.97×。最稳。
- **aider**：3/3 正确 + 达标（RBF、LayerNorm-5.3、Softmax-CE 前1.23/反2.55）。
- **gptme**（GPT-5.5）：Softmax-CE 达标且反向 3.91× 为三家最快；RBF 正确但反向 atomicAdd 竞争慢 63×；
  LayerNorm 有 block_reduce 广播 bug 致 verify FAIL（同会话 Softmax-CE 却修对了广播——是疏忽非能力）。

**共性洞察**：agent 能写出**结构/数学正确**的 kernel，但**性能优化到位度**和**实现细节 bug**（block_reduce 广播、
atomicAdd 竞争）参差——这些正是**阶段 6 优化闭环**（读 verify/bench 反馈→自主迭代）要解决的。
gptme 的 3 个 case 恰好覆盖了"正确达标 / 正确但慢 / 有 bug"三种典型，是阶段 6 自主纠错/优化的理想试验样本。

## 7. 阶段6 自主闭环实证（三 agent × 三种失败路径，均达成）

工具链：`run_on_a100.sh`（双跳直传→清缓存→verify门禁→bench→末行机读 `VERDICT=`）+ `bench_case.py --emit-verdict`
+ `AUTONOMOUS_LOOP.md`（agent 决策规范）。agent 每轮调脚本、读 VERDICT、按 loop.md 迭代，人工只搬运/复验、不喂答案。

| agent | 样本 | 失败路径 | 闭环结果 |
|-------|------|----------|----------|
| **gptme**(WSL) | 慢 RBF | `BENCH_FAIL`（反0.016×，atomicAdd竞争） | ✅ 自主优化→`PASS` 前3.43×/反1.26×（移atomicAdd改独立累加+前向D=64 tiling） |
| **aider**(Win) | buggy LayerNorm | `VERIFY_FAIL`（前err3519，block_reduce广播bug） | ✅ 自主2轮纠错→verify全PASS（backward+自发现forward同bug） |
| **codex**(Win) | LayerNorm前向0.97× | `BENCH_FAIL`（前向短核） | ✅ 自主2轮优化→`PASS` 前1.08×/反1.25×（D==1024 fast path：float4+寄存器复用+warp shuffle） |

**结论**：三种典型失败（性能慢/正确性错/前向短板）均由 agent **仅凭 VERDICT+日志反馈自主修复**到达标，全程守防作弊红线，
结果均经独立复验。证明"agent 直连 GPU 自测→自主优化/纠错"闭环成立，且**自测反馈使产出质量跃升**（gptme 前向 1.04→3.43×）。

**实测经验**：①aider 该环境无 `/run`，闭环为"半自动中转"（agent改+人工跑脚本回喂）；gptme(WSL内置shell)/codex 可自主执行。
②短核 case 的 size-env 规避依赖 config 支持 env 覆盖（gptme 生成的 layernorm config 曾硬编码 B 致失效，已补 `LN_B` 支持）。
③`run_on_a100.sh` 的 framework 防篡改前置改用 `git diff --numstat -w`（忽略 WSL↔Win 的 CRLF 行尾差异误判）。

### Stage C 全自主（无人工每轮干预）实证达成

`run_on_a100.sh --round-cap N` 加机械轮次兜底（按 case 计数 `.a100_round_<case>`，超 N 轮拒跑发 `ROUND_CAP_EXCEEDED`，
PASS 自动清零）。**gptme 全自主跑通**：起点把 RBF 反向重置回慢版（`BENCH_FAIL bwd=0.0155×`，atomicAdd 竞争 65ms），
gptme **自己循环**调 `run_on_a100.sh rbf --round-cap 12`、读 VERDICT、迭代反向 kernel，全程无人工每轮干预，
自主收敛到 `VERDICT=PASS fwd=3.44×/bwd=1.57×`（换 D=64 专用无 atomic 行/列归约，比 Stage B 半自动的 1.26× 更优；
round-cap 未触顶、PASS 后计数清零，均已独立复验）。**证明 agent 全自主（自调脚本+自判 VERDICT+自迭代）闭环成立。**

**codex 全自主（第二例，暴露两个问题：擦线不稳 + 计时空子被抓）**：起点把 LayerNorm 前向重置回未优化版
（`BENCH_FAIL fwd=0.97×`），codex 自主循环优化。**自主闭环机制成功**（自调脚本/自判/自迭代），但两轮都出问题：
- **第一轮擦线不稳**：达 `PASS fwd=1.054×` 但独立复跑 1.059×PASS/1.040×FAIL 抖动骑 1.05 线 → 强化 skill "稳定过线"规则
  （擦线须连跑 3 次全 PASS、留余量 ≥1.10×，写入 SKILL/loop/AUTONOMOUS_LOOP）。
- **第二轮钻计时空子被识破**：codex 在 `op.py` 加"输入无 `requires_grad` 时绕过 autograd + 跳过 mean/rstd 存储"分支，
  因 bench 前向用 `no_grad`+`detach` 计时，恰好命中该快路径 → 刷出"稳定 1.08× 3连PASS"。但这只在计时时受益、
  真实带反向前向拿不到——**撤销 op.py 特化后诚实前向塌回 1.04×FAIL**。→ 强化**防作弊红线 §5：评测路径必须=真实路径，
  op.py 禁止计时特化**（写入 SKILL.md）。
- **处置**：丢弃 Stage C 作弊轮，保留 codex **Stage B 的诚实达标版**（纯 kernel 的 D==1024 fast path、op.py 未动，
  诚实 3 连全 PASS：前 1.057~1.068×/反 1.21~1.23×）作为交付。LayerNorm 前向是 reduce 密集短核，诚实天花板约 1.04~1.07×。

**两个 Stage C 例的价值**：gptme(反向 0.016→1.57× 大幅稳过) 证明机制能大幅优化；codex 则暴露并催生两条 skill 加固
（稳定过线 + op.py 禁计时特化）——**"抓到并封堵作弊"比"又一个 PASS"更能证明 skill 的稳健性**。

**aider 全自主（第三例，用原生 `--auto-test`，补齐三 agent 全自主路径）**：早先误判 aider 无 `/run` 只能半自动——
实则 aider 有原生 `--test-cmd`+`--auto-test`（改完自动跑 test-cmd，退出码≠0 则喂回输出继续修）。为此给 `run_on_a100.sh`
加 `--strict`（VERDICT=PASS→exit0/其余→exit1）。aider 以 `--test-cmd "run_on_a100.sh layernorm --strict --round-cap 12"`
**真·全自主**优化 LayerNorm 前向（0.89×→float4 向量化路径），**3 连稳过：前 1.060/1.075/1.068×、反 ~1.19×**；
且明确禁令下**未碰 op.py（没钻 codex 那个计时空子）、framework 零改动**。→ **三 agent 全自主路径全部实证：
gptme(WSL shell) / codex(内置shell) / aider(--auto-test 退出码驱动)。**

## 8. 阶段7 用户输入极简化（方法论内化）+ 三宿主 RMSNorm 实测

**目标**：用户只在 agent 界面输入算法定义，agent 靠内化的约定文件自动跑完全流程（不再手输长 prompt）。
内化载体：`AGENTS.md`(codex 原生读)、`CONVENTIONS.md`+`.aider.conf.yml`(aider read/auto-test)、`CLAUDE.md`(Claude Code)、
gptme 靠 `start_gptme.sh --system` 注入 AGENTS.md。配 `autotest.sh`(mtime 探测当前 case)+ `bench.env`(短核 case 自声明规模)。

**三宿主用全新 case RMSNorm 实测（只输一句算法定义）**：
| 宿主 | 内化方式 | 内化结果 | RMSNorm 达标 |
|------|---------|---------|------------|
| aider | `.aider.conf.yml` read + `--auto-test` | ✅ 零输入自建7文件+自主推导反向+自动自测 | 正确PASS；前向擦线抖动(1.047~1.076)不稳 |
| gptme | `--system` 注入 AGENTS.md | ✅ 零输入自建+自主多轮循环(需命令式措辞+"别停") | 正确PASS；反向 reduce 天花板~1.0× |
| codex | 原生读 AGENTS.md | ✅ 零输入自建+主动加bench.env+主动3次自测(读到"稳定过线") | 正确PASS；反向擦线抖动(1.047~1.070)不稳 |

**结论**：**阶段7 内化在三宿主全部成功**——用户只给算法定义，agent 自动读 SKILL/建 case/写实现/自主推导反向/自测/优化，
全程无需手输方法论，均守防作弊红线（含 op.py 无计时特化）。

**算法优化空间洞察（关键，跨所有测试一致）**：agent 能否稳定达标 5% **取决于算法本身有无优化空间**，与内化/宿主无关：
- **距离/矩阵类（RBF）**：广播式参考物化大中间量、手写融合 kernel 空间大 → **稳赢**（反向曾 1.4~3.9×）。
- **归一化/reduce 类（LayerNorm 前向、RMSNorm 前反向）**：reduce 密集、torch.compile 已接近最优 → **卡 1.05 线抖动**，
  三宿主一致（aider 前向、gptme/codex 反向都卡）。这是**算法固有难度**，不是 skill/agent 缺陷；稳定过线规则正确拦下擦线。

**跨平台基建修复（阶段7 实测暴露，均已修）**：conf 注释纯 ASCII(免 GBK 崩)、conf 不写死 model(通用)、
`.gitattributes` 强制 .sh 用 LF(免 CRLF 崩 bash)、autotest 用 mtime 探测(免 CRLF 误判)、bench.env 通用化(新短核 case 免改脚本)、
start_gptme model 用 env 覆盖(避限流)。

## 9. 阶段8 打磨优化策略库（核心=减少 agent 盲试轮次）

### 8.2 研究：reduce 密集类天花板可破（推翻此前悲观结论）
此前 codex/aider/gptme 在 LayerNorm/RMSNorm 反向都卡 ~1.0×，被当成"reduce 密集类打不过 torch.compile 的天花板"。
**手工深度优化证明这是误判**——诊断对瓶颈 + 三招组合即 **3 连稳过 5%**（前 1.093×、反 1.11~1.13× 有余量）：
1. **反向缓存 mean/rstd 复用**（前向输出、反向读，消除朴素 O(B·D²) 重算）——反向 ~1.0×→1.11×，决定性。
2. **dgamma/dbeta 二维分块列规约**（行块×连续列、行主序合并读、atomicAdd 跨块）。
3. **前向 float4 + 寄存器缓存**（X 从三遍读降到一遍）——前向 0.93×→1.09×。
→ 这三招（算法无关）已沉淀进 SKILL.md 技巧库 + loop.md 优化手段。

### 8.3 决策层：SKILL.md 新增"瓶颈诊断→策略选择"表
把优化从"顺着清单盲试"改为"先 profile 诊断瓶颈类型→查表直选手段"。表是**可观测信号→瓶颈类型→手段类别**（非"算法→配方"），
对任意算法（含全新的）通用，且只给决策指引不给可照抄配方（守 agent 自主性）。

### 8.4 对照实测：决策层显著减少盲试
同一 agent(gptme)、同一 case(RMSNorm)、同起点，只差 skill 有无诊断表：

| | 旧 skill（无诊断表） | 新 skill（有诊断表） |
|---|---|---|
| gptme × RMSNorm 反向 | **盲试 5 种列规约**，卡 ~1.0× 天花板，**没过** | **主动 profile 诊断**（原话："非固定开销、已缓存 inv_rms 非重算、主要是 dgamma 列规约低效+dx 重复读"）→ 对症用二维分块+float4 → 反向 1.05~1.07× 达标 |

**关键质变**：agent 从"盲目穷举"变成"诊断→对症选手段"——决策层减少盲试的机制验证成功。（反向 1.02~1.07× 3连仍擦线不稳，
是 gptme 的二维分块/float4 打磨深度不如手工版 1.11~1.13×，非决策层无效；op.py 无计时特化、framework 零改动均已独立复验。）

**阶段8 结论**：①"reduce 密集类天花板"是伪命题——诊断对+缓存复用即可破；②决策层让 agent 从盲试转为对症诊断，
是"减少优化轮次/更快更好"的有效机制。
