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

**算法优化空间洞察（阶段7 判断，阶段8 部分修正）**：agent 能否稳定达标 5% 取决于算法优化空间 + **是否诊断对瓶颈**：
- **距离/矩阵类（RBF）**：广播式参考物化大中间量、手写融合空间大 → **稳赢**（反向曾 1.4~3.9×）。
- **归一化/reduce 类（LayerNorm/RMSNorm）**：阶段7 时三宿主都卡 1.05 线、误以为"reduce 密集固有天花板"；
  **阶段8 推翻此判断**——诊断对瓶颈 + 缓存 mean/rstd 复用 + 二维分块列规约 + 前向 float4 即可稳过（8.2 手工版前1.09/反1.11~1.13）。
  卡线的真因是**没诊断对+没上缓存复用**，非算法固有天花板。稳定过线规则仍正确拦擦线。

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

### 8.4 对照实测：决策层减盲试 + 纯输入暴露盲区 + skill 迭代见效（三宿主）

同 case(RMSNorm)、**纯算法输入**（不喂任何方法论/优化/流程提示，方法论全靠内化约定）：

| 宿主 | skill 版本 | 建 bench.env? | 被短核假象骗? | 独立复验（放大规模）真实结果 |
|------|-----------|--------------|--------------|--------------------------|
| gptme | 决策表版 | ✓ | 否 | **主动诊断**"dgamma列规约低效+dx重复读"→对症二维分块+float4；反向 1.02~1.07× 擦线抖动 |
| aider | 决策表版 | ✗ **漏建**(config写死B=4096) | **是**(误信短核 前1.22/反1.43 PASS) | 放大后真实 **前0.99 FAIL/反1.07** |
| codex | **+短核警惕强化版** | ✓ **主动建**(RMS_B=22528) | 否 | **前1.12~1.14/反1.09~1.13，3连稳过有余量**（三宿主最干净） |

**三层递进发现**：
1. **决策层减盲试**（gptme）：从"盲试5种列规约卡死"→"profile诊断对症选手段"，机制验证成功。
2. **纯输入暴露 bench.env 盲区**（aider）：漏建 bench.env → 被短核固定开销虚高骗（1.22×假象）→ 误判 PASS 收工。
   这是"输入只给算法定义"原则的价值——不喂方法论才暴露 agent 真实内化盲区。
3. **skill 迭代见效（codex）**：据 aider 盲区**强化 skill**（AGENTS.md/SKILL.md 加"短核假象警惕：baseline<0.15ms却高加速比→
   放大规模重测"）后，codex 用强化版**主动建 bench.env、没被骗、3连稳过有余量**——"实测暴露盲区→改 skill→再测见效"闭环。
4. **强化吸收差异 = agent 架构行为，非模型能力（aider 5.4→5.5 重测）**：aider 用同一强化版重测 RMSNorm——
   config 支持了 env（吸收一半），但**仍漏建 bench.env、仍被短核假象骗**（短核显 前1.2/反1.3~1.4 PASS，放大后真实
   5.4 前0.99/反1.08、**5.5 前0.96/反1.00 双 FAIL**）。**换更强模型 5.4→5.5 也没堵上**（config env 吸收更全，但"建 bench.env"
   "前向上 float4"两步俩都漏）→ 盲区是 **aider 壳的行为**（偏向只编辑列进 chat 的文件，不主动创建约定提到的额外文件/不深挖优化），
   **非模型能力**。对比 codex（内置 shell、主动性强）用同一强化版就主动建 bench.env、稳过。**agent 架构（自主性）是独立于 skill 质量和模型能力的第三维。**

### 8.1 扩新形态 case：前缀和（scan）验证决策层对全新算法通用
codex 用**纯算法输入**（只给"y[b,i]=Σ_{j≤i}x[b,j]，X[4096,4096]，对 X 求梯度"）从零建 scan case——这是现有三形态
（距离 RBF / 归一化 LayerNorm·RMSNorm / 损失 Softmax-CE）之外的**第四种形态：累积/扫描依赖**。结果：
- **自主推导反向**（前缀和反向=反向扫描 `dx[j]=Σ_{i≥j}dy[i]`），正确性 5 种子全 PASS。
- 用 **CUB block scan**（标准并行扫描原语，非 torch 高层落回，合规），**前 3.43×/反 4.13× 大幅达标**（非短核，独立复验）。
- → **决策层/skill 对全新算法形态通用**：codex 没卡在"没见过的形态"，照样诊断+实现+达标。scan 属**易赢类**（并行扫描优化空间大，
  像 RBF 一样大幅超线），再次印证"算法优化空间决定达标难度"。据此给决策表补一行"累积/扫描依赖→用成熟并行扫描原语(CUB)"。

**阶段8 结论**：①"reduce 密集类天花板"是伪命题——诊断对+缓存复用即可破（8.2 手工版前1.09/反1.11~1.13 稳过）；
②决策层让 agent 从盲试转对症诊断（减轮次）；③纯算法输入是暴露内化盲区的关键手段，据此迭代 skill 能堵漏，
但**吸收程度依 agent 架构而异**（同一"短核警惕"强化：codex 主动建 bench.env 避坑；aider 换 5.4/5.5 均漏——是壳的自主性弱，非模型能力）；
④决策层对**全新算法形态通用**（8.1 scan 大幅达标）。
**三维独立**：skill 质量 / 模型能力 / agent 架构自主性——改 skill 只能提升第一维。op.py 无计时特化、framework 零改动全程独立复验。

## 10. harness 侧兜底适配 aider 架构弱点（--auto-scale）

8.4 发现 aider 架构弱点（换 5.4/5.5 均如此）：不主动建 bench.env → 短核 case 被固定开销虚高骗 → 误信假 PASS 收工。
改 SKILL 文档堵不上（架构自主性维度）。故在 **harness 侧兜底**：`run_on_a100.sh --auto-scale`（默认开）——
无 size-env/bench.env 时，grep config 的 `os.environ.get` 发现规模 env 变量，远程先默认规模探测 baseline 耗时，
任一 <0.15ms 判短核 → 用该 env=32768 放大重测，VERDICT 基于真实数。显式 size-env/bench.env 仍优先；非短核不误伤。

**端到端验证（aider，纯算法输入，不建 bench.env）**：
| | 旧脚本 | auto-scale 兜底 |
|---|---|---|
| 短核假象 | 被骗（短核假 PASS） | 自动放大→真实数，不被骗 |
| aider 收到信号 | 假 PASS→exit0→**误停** | 真实 FAIL→exit1→**继续优化** |
| 结果 | 停在假达标（真实前0.99） | 被驱动优化：前向真达标 1.09、反向从~1.0 逼到 1.04（3连仍<1.05，差一档） |

**结论**：harness 兜底把较弱宿主也拉上**诚实优化闭环**——aider 即便不建 bench.env 也不再被短核假象骗停，
拿到诚实 FAIL 后真的继续优化（前向从假象背后的真实 0.99 优化到真达标 1.09）。反向 1.04 差一档没稳过是 aider 
优化深度上限（架构自主性维度，非兜底失败）。**测量鲁棒性归 harness、优化深度归 agent**——职责分离后，
skill/harness 能保证的（诚实信号）与 agent 能力决定的（优化到什么程度）清晰分开。不改 aider 本体，宿主无关。

## 11. 扩形态：GEMM+bias+gelu 融合（三宿主，暴露"硬 case 宿主能力分层"）

第五种形态：GEMM 融合类（Y=gelu(X@W+b)，tanh 近似）。**每个新形态都三宿主验证**（用户要求）。纯算法输入。

| 宿主 | 结果 | 手法 |
|------|------|------|
| **aider**(GPT-5.5) | ✅ 前1.72/反1.22 | cuBLAS SGEMM + 手写 float4 bias+gelu 融合尾 |
| **codex**(gpt-5.6-sol) | ✅ 前1.19/反1.16 | cuBLAS + float4 融合 + 融合 dZ/db；**主动 cublasSetMathMode(DEFAULT) 禁 TF32 保 fp32 精度**（最严谨） |
| **gptme** | ❌ 未拿下 | 5.4：正确但 cuBLAS 用得同 baseline、加速比 1.00× 无融合优势；5.5：畏难"无法继续"空转；GLM-5.2：reasoning 空转推 wmma 到 context 耗尽不落文件 |

**新红线判定（cuBLAS/CUB 允许）**：GEMM 用 `cublasSgemm` 合规——cuBLAS/CUB 是 CUDA 官方底层库、非 torch 高层算子（红线禁的是 `F.*`/SDPA/`torch.matmul`）。赢 torch.compile 靠**融合**（bias+gelu 融进 GEMM 尾、省中间物化/额外 launch），是正当优势。已写进 SKILL 红线 §1。**注意 TF32**：cuBLAS 在 A100 默认对 SGEMM 用 TF32（降精度），须显式 `CUBLAS_DEFAULT_MATH` 关掉守 fp32——codex 做对了。

**硬 case 宿主能力分层**：GEMM 这类 cuBLAS 强项，只能靠融合优势险胜，需要 agent ①想到用 cuBLAS+融合 ②正确实现 cuBLAS 句柄/mathmode ③做出融合。GPT-5.5(aider)/gpt-5.6(codex) 都做到；gptme 配的模型都够不到（5.4 会用 cuBLAS 但没融合优势、5.5/GLM 根本没产出）。**印证三维独立**：同样纯输入+同 skill，硬 case 上模型能力/agent 架构（aider --auto-test 退出码驱动、codex 内置 shell 强自主 vs gptme 手动 shell+易空转）决定成败。

**并发踩坑（血泪）**：多个 run_on_a100 同跑同一 case（尤其 aider --auto-test 未关一直发任务 + 人工并行复现）→ torch file_baton 编译锁竞态 + 缓存目录 `rm -rf` 互撞 + 残留僵尸 bench 进程占锁 → 一连串假 VERIFY_FAIL（非代码问题）。**教训**：①同一 case 串行跑、勿并行；②agent auto-test 跑完务必关闭（别留后台循环）；③排查 VERIFY_FAIL 先查远程 `ps aux|grep bench_case` 有无残留僵尸，`pkill -9 -f <case>` + 清缓存再复验。

## 12. 扩形态：online-softmax（三宿主，暴露第二类基线陷阱"弱 baseline 假象"）

第六种形态：attention 核心子结构——单遍在线 softmax（running max/sum，遇更大值按 exp 差修正已累加 sum）。纯算法输入。

| 宿主 | 结果 | 手法 / 说明 |
|------|------|------|
| **aider**(GPT-5.5) | ✅ 前1.11/反2.92 | 手写在线扫描 kernel、**向量化 reference**（torch.compile 好编、强 baseline）、C=1024 float4 路径 |
| **codex**(gpt-5.6-sol) | ✅ 3×PASS 前~1.10/反~2.95 | 同上，向量化 reference、op.py 干净、bench.env `ONLINE_SOFTMAX_B=32768` |
| **gptme**(GPT-5.4) | ⚠️ verify PASS、bug 自修，但 **reference 写成 1024 次 Python 逐列 for 循环** → bench 卡死；换诚实向量化 baseline 后**前0.92 FAIL**/反1.63 | 代码合规（op.py 无特化、防作弊守）、返回类型 bug（`return probs`→`return {probs,row_max,row_sum}`）**自主修对**、candidate 前反向反复调用飞快（10次各0.01s）；**败在 reference/前向 kernel** |

**gptme 诊断全过程（关键，第二类基线陷阱）**：
1. gptme 建全 8 文件、合规，返回类型编译 bug 自测拿到报错后**自主修对**（花括号构造 vector）。
2. bench 却卡死在 `case=` 行超时——根因**不是死锁/代码错**，是 gptme 把算法描述"单遍在线扫描"**逐字翻译成 `for c in range(1,C): ...` 1024 次 Python 逐列循环**当 reference。`framework/bench.py` 对 reference 做 `torch.compile` 作 baseline → 展开 1024 次循环成 O(C) 巨型图 → **首次编译 44.65s**（实测）+ 反向编译 → 累计超时卡挂。
3. 期间多次假 VERIFY_FAIL 是**复验并发自伤**（SSH timeout 断开不杀远程子进程 → 叠加 30+ 僵尸 bench → ninja 锁竞态 `can't create .o`），非 gptme 代码问题。清僵尸+串行单跑后 verify **全 PASS**。
4. **坐实"弱 baseline 假象"**：把 gptme 的 reference 改成向量化等价（`exp(x-amax)/sum`，数学恒等、verify 仍全 PASS 且误差更小 7e-9），重跑 bench → **前0.9155× FAIL**/反1.625× PASS。即：gptme 原逐列 baseline 畸形慢（eager 每次0.069ms），candidate 跟它比会**虚高**；换诚实向量化 baseline，gptme 前向真实**打不过** torch.compile（0.2585ms vs baseline 0.2366ms）。

**第二类基线陷阱（继"短核假象"后）——"弱 baseline 假象"**：reference 写成 Python 逐元素/逐列循环 → ①`torch.compile` 编译爆炸卡死 bench；②eager baseline 畸形慢、加速比虚高不诚实。**根因是 agent 把描述的"串行叙述（单遍/在线/逐步）"当实现方式字面翻译，而非识别其数学语义**。aider/codex 会主动向量化（识别出等价标准 softmax），gptme 字面翻译——又一个**三维独立**佐证（模型/架构层面 gptme 更倾向字面直译）。

**skill 已强化（纯输入暴露盲区→改 skill，同"短核假象"闭环）**：SKILL.md 步骤2 + AGENTS.md/CLAUDE.md/CONVENTIONS.md 加约定——**reference 必须向量化，禁 Python 逐元素/逐行/逐列 `for` 循环**；"单遍/在线扫描/逐列累加"是数学语义不是实现方式，用广播+整体规约表达；自检：reference 里出现 `for` 遍历张量维度几乎一定写错。

**online-softmax 前向本身难赢**：torch.compile 的向量化 softmax 前向已高度优化，前向达标要 candidate kernel 真做出访存优势（aider/codex 靠 float4 险胜 1.10~1.11×）；gptme 前向 kernel 不够优，诚实基线下 0.92× 打不过。反向（softmax 雅可比 `dx=P*(dy - sum(P*dy))` 融合）三宿主都大幅赢（1.6~3.0×），是稳赢区。

## 13. 扩形态：RoPE 旋转位置编码（三宿主全通，向量化约定生效验证）

第七种形态：LLM 注意力位置编码——RoPE。输入 X[B,S,H,D] fp32，最后一维按相邻二元组做 2D 旋转，位置 s、维度对 i 的角度 `θ = s·base^(−2i/D)`。纯自然语言算法输入（不预放 description.md，agent 自建整个 case 目录）。

| 宿主 | 结果 | 手法 / 说明 |
|------|------|------|
| **gptme**(GPT-5.4) | ✅ 前2.54/反1.18 | reference **主动向量化**（注释里复述"必须向量化"约定！）、纠正"本地无 torch 不能自测"认知盲区后**自主诊断+修 kernel bug**：初版 `s_idx = row % S`（错误位置索引） → err~11 → 自修 `s_idx = (row/H) % S`（B*S*H 扁平后 S 在中间维）；建了 bench.env；反向合理省 `save_for_backward(x)`（RoPE 反向只需 grad_out + 固定角度，不需原始 X） |
| **aider**(GPT-5.5) | ✅ 前1.15~1.24/反2.42~2.56 | 向量化 reference、**主动建了 bench.env**（`ROPE_B=32`）——RMSNorm 时死活不建的架构弱点这次没犯；擦线抖动：#1 fwd 1.06 擦线、#2 环境错（A100 /tmp 撞满 `No space left on device`，别人的 CUTLASS 编译产物撑爆）、#3 fwd 1.15；两次干净 PASS 都 >1.10 判达标 |
| **codex**(gpt-5.6-sol) | ✅ 前1.51/反2.21 | 向量化 reference、**float4 快路径 + float2 通用回退**（D=64/128 时走 float4）、`sincosf` 同求正余弦、反向应用旋转矩阵转置、bench.env `ROPE_B=16 S=1024 H=16 D=64`；数值最干净（err~3.5e-5，比 gptme 6e-4/aider 4.8e-7 都合理）；一贯的高质量 |

**本轮关键发现 1——向量化约定跨宿主生效**（继"短核假象"后第二次纯输入闭环成功）：online-softmax 时 gptme 因字面直译"单遍扫描"写成 Python 逐列循环踩了"弱 baseline 假象"→ 强化 SKILL/AGENTS/CONVENTIONS/CLAUDE 加"reference 必须向量化"约定 → 本轮 RoPE **三宿主全部主动向量化**（gptme 甚至在注释里复述约定"必须向量化，不使用 Python 沿张量维度的 for 循环"）。**证实"纯输入暴露盲区→改 skill→再测见效"完整闭环**，与阶段8.4"短核假象"同款成功。

**本轮关键发现 2——aider 主动建 bench.env**：aider 架构弱点（不主动创建约定提到的额外文件）在 RMSNorm 反复暴露（GPT-5.4/5.5 都栽），harness 已 `--auto-scale` 兜底。这次 RoPE aider 却**主动建了 bench.env**——可能 skill 反复强化开始渗透，或 RoPE 多维张量 [B,S,H,D] 让"需放大规模"的信号更明显（4 维乘起来天然大）。**未定论、需更多 case 观察**：这是 skill 渗透进 aider 架构的迹象，还是 case 特点使然？下轮扩形态注意 aider 是否继续主动建 bench.env。

**本轮关键发现 3——三宿主稳赢形态**：RoPE 和 RBF/scan 一类，可优化空间大（访存密集 elementwise、baseline torch.compile 因 D 维小做不出多少融合），三宿主全通。对比：LayerNorm/RMSNorm reduce 密集擦线抖动、GEMM+bias+gelu 卡宿主能力分层（gptme 挂）、online-softmax 前向卡宿主（gptme 弱 baseline 假象暴露前 verify 全过、修 baseline 后 0.92× 打不过）。**七形态覆盖（RBF/LayerNorm/softmax_ce/scan/GEMM/online-softmax/RoPE）已呈现完整光谱**：稳赢区（RBF/scan/RoPE）、擦线区（LN/RMSNorm 前向、reduce 密集）、宿主分层区（GEMM/online-softmax 前向）。

**collected as 第六形态参考 case**：codex 版数值最干净 + float4 优化最完整，收进 `cases/rope/` 作参考。

**干净房间踩坑（记进 [[feedback-new-case-three-hosts]]）**：aider 首次启动报 `litellm.BadRequestError: 模型服务调用失败`——fresh clone 丢了 gitignored 的 `.aider.model.settings.yml`（`use_temperature: false`，治 GPT-5.5 撞代理 temperature 的老坑）。手动从旧 workdir `cp` 过来即恢复。**教训**：每次新建 aider 干净房间必须补机器特定本地文件（settings、SSH 密钥），它们 gitignore 不带过来但宿主起不来会依赖它们。

## 14. 扩形态：线性 SSM（时序递推 scan 家族，三宿主暴露"弱 baseline 假象的第二种变种"）

第八种形态：真时序依赖的线性状态空间模型——`h_t = a·h_{t-1} + b·x_t`（a=0.9、b=1.0 标量常数）、输出 `y_t = h_t`。**纯自然语言算法输入**，无预放 case 文件。

**关键考验**：SSM 是**真时序依赖**（前一步输出=后一步输入，不能整体向量化成单个广播），reference 该如何写既遵守向量化约定又数学正确？正解是**数学变换转成 O(T) 前缀和**：`h_t = a·h_{t-1}+b·x_t ⟹ h_t = b·a^t·cumsum(x/a^t)[t]`，用 `torch.cumsum` 表达。

| 宿主 | 结果 | 手法 / 说明 |
|------|------|------|
| **codex**(gpt-5.6-sol) | ✅ 前3.22/反4.81 | 主动想到 cumsum 数学变换（`b·powers·cumsum(x·inv_powers)`，用乘法代替除法数值更稳）、bench.env `LSSM_B=2048`；首次因用户工作目录未设对建到临时目录（用户澄清后重测），重启后**一次到位**建到 `cases/linear_ssm/`、一次 PASS 无迭代 |
| **aider**(GPT-5.5) | ✅ 前3.11/反4.46 | 同款 cumsum 变换（`b·powers·cumsum(x/powers)`）；未建 bench.env 但 baseline 0.67ms 非短核不需要；一次 PASS |
| **gptme**(GPT-5.4) | ⚠️ **BENCH_INVALID**（3 版都踩弱 baseline） | **第 1 版**：`for t in range(T): h = a*h + b*x[:,t,:]` 时序循环 → torch.compile 编译展开 1024 次时序 for → bench 死锁 fork **30+ bench_case 进程炸弹** → 拿不到 VERDICT；**第 2 版**（未纳入统计）：无实际差异；**第 3 版**（拿到 44×/54×）：改成 `torch.tril(W) + einsum("tk,bkc->btc", W, X)` 密集 T×T Toeplitz 矩阵——**伪向量化**，把 O(T)=1024 算法恶化到 O(T²)=1M，baseline 慢在算力（O(B·T²·C) 密集 matmul） → **44×/54× 加速比虚高**、与真 O(T) baseline 下 codex/aider 的 3~5× 完全不可比；判 BENCH_INVALID |

**gptme 三版失败诊断（关键，"向量化"约定的深度缺口）**：
1. **v1**（`for t in range(T)`）：gptme 把约定"必须向量化"理解成"避免逐元素循环"，认为时序 for 是数学语义合法。**踩的坑**：SKILL/AGENTS 原文只写"逐元素/逐行/逐列"没提"时序/序列"，边界模糊。
2. **v3**（`torch.tril + einsum`）：纠正约定含义后 gptme 改到 T² Toeplitz——它把"向量化"理解成"用张量算子表达"，没意识到**算法复杂度必须与最优实现一致**。**踩的坑**：SKILL 原文没禁"用 O(N²) 密集矩阵代替 O(N) scan/cumsum"这个高级伪向量化。
3. 每次 verify 都 PASS（数学都对）但 bench 层面失败（编译爆炸/弱 baseline）——**verify 只查数学、bench 才暴露评测层坑**。

**弱 baseline 假象的第二种变种（继"字面 for 循环"后的高级变种）**：
- **第一种变种**（online-softmax gptme 首次）：Python for 循环 reference → torch.compile 编译爆炸 + eager 畸形慢 → 加速比虚高
- **第二种变种**（linear_ssm gptme v3）：**算法复杂度错误的向量化**——用 O(N²) 密集矩阵（如 `torch.tril + einsum`）代替 O(N) scan/cumsum → baseline **慢在算力**（正确 baseline 需要"并行 scan"，但 gptme 写成 dense matmul,torch.compile 无从推断三角结构还原成 scan） → 加速比虚高

**skill 已再强化**（本轮又一次"纯输入暴露盲区→改 skill"闭环）：SKILL/AGENTS/CONVENTIONS/CLAUDE 四份文件同步扩展"向量化"约定——
- **明确"任何张量维度的 Python for 循环"**（包括时序/序列维度，不只是"逐元素/逐列"）
- **新增"算法复杂度必须与最优实现一致"约定**：scan/递推类禁 `torch.tril + einsum` 的 O(T²) 密集矩阵变体，要用 `torch.cumsum` 类 O(T) 前缀原语
- 给出 SSM 的具体数学变换公式作正例

**三宿主模型能力分层再次显现**（与 GEMM+bias+gelu §11 同款结构）：
- codex(gpt-5.6-sol)/aider(GPT-5.5)：**主动想到数学变换**（`h_t = b·a^t·cumsum(x/a^t)`），一次到位
- gptme(GPT-5.4)：**能识别约定但找不到正确变换方向**——被"数值上溢下溢"担忧拉去 for 循环，被"避免 for 循环"约束拉去 T² Toeplitz——**看到坑但绕不过**
- **不是内化不足，是模型能力上限**：gptme v3 后 skill 反复强化约定也不会让 GPT-5.4 想到 cumsum 变换（这是数学推导能力）。**印证[[feedback-new-case-three-hosts]] 三维独立**：模型能力(gpt-5.4 vs 5.5 vs 5.6)、宿主架构、skill 质量三维互不相同。

**八形态光谱**：稳赢区（RBF/scan/RoPE/linear_ssm cumsum）、擦线区（LN/RMSNorm 前向 reduce 密集）、宿主/模型分层区（GEMM/online-softmax 前向、linear_ssm gptme 弱 baseline）。

**collected as 第七形态参考 case**：codex 重测版收进 `cases/linear_ssm/`（cumsum 变换 + 乘法代除法数值更稳 + bench.env LSSM_B=2048）。

**过程踩坑**：本轮多次撞**并发编译锁竞态**——我在诊断 gptme 时不该在同一远程跑 verify（我的 verify 与 gptme 的 run_on_a100 争 file_baton 锁 → 一堆假 VERIFY_FAIL），教训与 GEMM 时同款：**agent 自主自测跑起来后我要保持不动**，否则 SSH timeout 断开不杀远程子进程 → 叠加进程炸弹 → GPU 占满 → 双方都拿不到 VERDICT。


## 15. 扩形态：Welford 单遍在线均值/方差 LayerNorm（三宿主，验证约定强化生效 + 暴露"弱 baseline 假象第三变种"）

第九种形态：Welford 数值稳定单遍统计——描述**故意用串行 recurrence**（`count_n=n; delta_n=x_n-mean_{n-1}; mean_n=mean_{n-1}+delta_n/n; M2_n=M2_{n-1}+delta_n·delta2_n`），得 mean/var 后做 LayerNorm（`y=gamma·(x-mean)/sqrt(var+eps)+beta`，对 X/gamma/beta 求梯度）。**纯自然语言输入**，无预放 case 文件。

**关键考验**：这是 online-softmax 同族的"单遍在线扫描"陷阱结构，且用了 linear_ssm 之后**已强化的向量化约定**（禁任何维度 for + 禁 O(N²) Toeplitz）。**正解**：识别 Welford ≡ 标准两遍统计 `mean=X.mean(-1); var=((X-mean)²).mean(-1)`（数学恒等），向量化 O(N)。本轮首要目的是**验证约定强化对踩坑户 gptme 是否生效**。

| 宿主 | 结果 | 前向 | 反向 | 手法 / 说明 |
|------|------|------|------|------|
| **codex**(gpt-5.6-sol) | ✅ 达标 | 1.097×(擦线3连) | 1.27× | 识别 Welford=两遍统计，向量化，注释明写"loop-free graph for torch.compile"（**理解约定意图**）；D=1024 float4 快路径+前向统计缓存；擦线独立复验 3 连稳过（1.0968/1.0996/1.0971） |
| **gptme**(GPT-5.4) | ⚠️ 未过（诚实达线未过前向） | 1.033× FAIL | 1.072× PASS | **约定强化生效**：reference **向量化两遍统计**（无 for/tril）、注释写"避免 Python 沿张量维度循环，便于 torch.compile 作为诚实 baseline"——**理解了为什么**；正确性+反向过线，**仅前向擦线 1.033 没抠够**（reduce 密集前向，模型优化力不及 codex 的 1.097） |
| **aider**(GPT-5.5) | ❌ 未过（弱 baseline 假象） | ~~2.71~~→**0.99** FAIL | ~~2.89~~→**0.91** FAIL | 表面 PASS 前2.71/反2.89，实为 **reference 藏"规模专属慢分支"**：`if x.numel()>=64M: 干净mean/var else: cumsum+一串中间张量慢路径`——bench 默认 16.7M 命中慢分支 → 弱 baseline。强制 `--size-env` 放大到 >64M 走干净分支后**真实仅 0.99/0.91 FAIL** |

**约定强化生效验证成功（本轮核心结论）**：gptme 是踩坑户（online-softmax for、linear_ssm v1 for + v3 T² Toeplitz），Welford 又是"单遍在线扫描"陷阱结构。**读强化约定后 gptme 这次主动向量化、写诚实 baseline、正确性+反向都过**——证明"纯输入暴露盲区→改 skill→再测见效"闭环对**弱模型也生效**（与 online-softmax 后 RoPE 生效同款）。gptme 前向 1.033 没过是**纯模型优化力不足**（reduce 密集前向 codex 都只 1.097），属三维中的"模型能力"维，**与约定无关**——干净的三维分离。

**弱 baseline 假象的第三种变种（aider，最隐蔽）**：
- 第一种（online-softmax gptme）：Python for 循环 → 编译爆炸 + eager 畸形慢
- 第二种（linear_ssm gptme v3）：O(N²) 密集矩阵伪向量化 → 慢在算力
- **第三种（welford aider）：reference 规模/条件专属慢分支** `if numel>=阈值:快 else:慢`——**表面无 for/tril，grep 全查不出**，bench 默认规模命中慢分支 → 弱 baseline。比前两种更隐蔽，只能靠读源码查 `numel/shape` 分支 + 手动放大复验坐实。

**skill 已再强化（第三次"纯输入暴露盲区→改 skill"）**：SKILL/AGENTS/CONVENTIONS/CLAUDE 四份同步新增**"reference 必须单一最干净向量化路径，禁止规模/条件专属分支"**约定（附 welford 反例：`numel<64M 走 cumsum 慢分支` 刷 2.71× vs 干净 0.99×）。

**harness 盲区（记录，待加固）**：`--auto-scale` 只兜底第一类短核，对弱 baseline 三变种（for/T²/规模分支）**全部无效**——auto-scale 放大规模时变种 B（规模分支）反而可能跳到快分支掩盖问题。加固方向：harness 加"reference 一致性交叉检查"（同 case 两个差异大规模跑，加速比随规模剧变→报警）。详见 [[project-baseline-traps]]。

**九形态三宿主对照的稳定规律**：codex（gpt-5.6，内置 shell 主动性强）几乎每形态都最干净稳过；aider（GPT-5.5，架构偏"只编辑列进 chat 的文件"）易漏建 bench.env、本轮更画蛇添足加规模分支自造弱 baseline；gptme（GPT-5.4，能力最弱）踩坑最多但**约定强化后能避开陷阱、产出诚实结果**，卡点退化为纯优化力不足。**三维独立**（模型能力/宿主架构/skill 质量）在九形态上反复印证。

**collected as 第九形态参考 case**：codex 版收进 `cases/welford/`（Welford=两遍统计向量化 reference + D=1024 float4 快路径 + 前向统计缓存 + bench.env）。


## 16. 扩形态：单头 causal self-attention（第十形态，transformer 真核心，宿主能力分层最极端——仅 codex 拿下）

第十种形态：单头 causal self-attention——`S=Q·Kᵀ/√d → causal mask(上三角-inf) → softmax → O=P·V`，对 Q/K/V 求梯度。**纯自然语言输入**，无预放 case 文件。这是**多算子融合链**（两个 GEMM + mask + softmax），组合了 GEMM（§11）与 online-softmax（§12）两形态能力，是最接近真实融合算子、baseline 最强的目标结构。

**关键考验**：①红线§1（attention 最诱惑落回 `F.scaled_dot_product_attention`）；②多算子融合（能否 fuse QKᵀ+mask+softmax+PV，flash-attention 思路省 O(T²) 中间物化）；③baseline 强度（torch.compile 对 attention 融合极强，能否赢是悬念）。

| 宿主 | 结果 | 前向 | 反向 | 手法 / 说明 |
|------|------|------|------|------|
| **codex**(dongcc GPT-5.5) | ✅ 达标 | 1.26~1.37×(3连) | 1.16~1.40× | **唯一拿下**：cuBLAS batched GEMM(QKᵀ+PV 两个矩阵乘) + 自定义 causal softmax/dS kernel + 保存 probs 供反向；不落回 SDPA；擦线 3 连独立复验稳过（自测报 1.07~1.10、我复验 1.26~1.37 更高）；组合了 §11 cuBLAS + §12 手写扫描能力 |
| **gptme**(GPT-5.4) | ❌ 未达标 | 0.41× | **0.027×**(慢37倍) | 合规（无 SDPA、reference `bmm+masked_fill+softmax+bmm` 诚实）、正确性全 PASS，但**朴素 kernel 性能惨败**；gptme **自己诊断对方向**（"需 block-tiled QK/online softmax/PV 融合，接近 FlashAttention，不是小修小补"）但**写不出**——模型能力上限 |
| **aider**(GPT-5.5) | ❌ 未达标 | 0.17× | 0.47× | 合规（无 SDPA、reference 手写数值稳定 softmax，诚实 baseline）、正确性全 PASS，但**朴素 kernel 打不过融合 baseline**（合理规模 B=16/T=384/D=64 下前向 1.24ms vs baseline 0.21ms）；未建 bench.env（老毛病）；与 gptme 同类诚实性能不足 |

**宿主能力分层最极端的一例（本轮核心结论）**：同一 case、三宿主 reference 全部合规诚实（都用基础算子、无 SDPA、无弱 baseline），差异**纯在 kernel 实现力**——codex 用 cuBLAS+融合前向 1.3× 稳赢，aider/gptme 朴素 kernel 前向 0.17/0.41× 惨败。**attention 是"越接近真实融合算子，skill 越难跨宿主稳赢"的最强证据**：skill 方法论到位（三宿主都知道要融合、gptme 甚至诊断出 FlashAttention 方向），但**补不上中弱宿主的 kernel 实现力**。印证[[feedback-new-case-three-hosts]]三维独立——这次是"模型能力"维拉开最大差距（GEMM §11 gptme 挂但 aider 用 cuBLAS 赢过；attention 更难，aider 也挂，只剩 codex）。

**harness 缺陷（本轮暴露，待加固）**：`--auto-scale` 探到短核后**固定放大规模 env 到 32768**——这对 element-wise/reduce 类合适，但对 **O(B·T²·d) 的 attention 类算法放大值严重不当**（B=32768 → candidate 前向 283ms，夸大差距）。auto-scale 应按算法复杂度/显存上限自适应放大，而非固定 32768。当前规避：对 attention 类用 `--size-env` 显式给合理规模（如 B=16/T=384/D=64）。记 [[project-baseline-traps]]。

**十形态光谱更新**：稳赢区（RBF/scan/RoPE/linear_ssm cumsum/Welford codex）、擦线区（LN/RMSNorm/Welford 前向 reduce 密集）、**宿主分层区（GEMM/online-softmax/causal_attn——融合密集，中弱宿主难赢，attention 最极端仅 codex 拿下）**。**规律**：算法的 kernel 优化空间越依赖"多算子融合/FlashAttention 级重写"，越考验宿主 kernel 实现力，skill 方法论指对方向但代替不了实现能力。

**collected as 第十形态参考 case**：codex 版收进 `cases/causal_attn/`（cuBLAS batched GEMM QKᵀ+PV + 自定义 causal softmax/dS kernel + 保存 probs 供反向 + bench.env B=16/T=384/D=64）。


## 17. 扩形态：depthwise causal conv1d（第十一形态，全新卷积家族，暴露 harness auto-scale 阈值边界缺陷）

第十一种形态：逐通道因果一维卷积——`y[b,c,t]=Σ_{k=0}^{K-1} w[c,k]·x[b,c,t-k]`（X[B,C,T]、核 W[C,K] K=4、因果左 padding、通道间不混合），对 X 和 W 求梯度。**全新卷积家族**（不属之前距离/归一化/scan/融合任何形态），滑窗访存、shared-mem tiling 优化空间大，中等难度。**纯自然语言输入**。

**关键考验**：①禁 `F.conv1d`（红线§1）；②"沿小固定核长 K=4 展开"是否被误判违反向量化约定（禁的是沿大数据维 B/C/T 循环，不是禁 K=4 这种小固定核长展开成 4 个广播项）。

| 宿主 | 结果 | 前向 | 反向 | 手法 / 说明 |
|------|------|------|------|------|
| **codex**(gpt-5.6) | ✅ 达标 | 4.78× | 1.86× | reference `x*w0+shift(x,1)*w1+...` K=4 手写展开（正确理解小核长展开合规）、无 F.conv1d；kernel shared-mem tiling + warp 规约 dW；放大规模 B=128/C=512/T=1024 一次 PASS |
| **gptme**(GPT-5.4) | ✅ 达标 | 1.78× | 3.72× | reference `unfold`+广播乘加向量化、无 F.conv1d、无 for；bench.env B=64/C=1024/T=4096；**建全合规代码但卡自测 thinking**（我代跑复验 PASS）——自主性弱（执行力），非能力问题。**对比 attention 惨败（0.027×）：conv1d 有明确 tiling 套路无需 FlashAttention 级重写，gptme 拿下** |
| **aider**(GPT-5.5) | ❌ 未达标 | 1.77× 真赢 | **短核假象** 1.32→0.96 FAIL | reference K=4 手写展开合规、无 F.conv1d；**未建 bench.env**（老毛病）→ 默认小规模（B=32/C=256/T=1024=8.4M）→ 反向短核虚高 1.32×，放大真实规模（B=64/C=1024/T=4096）后**反向真实仅 0.96× FAIL**（前向真赢 1.77×）。自测看到短核 PASS 被骗 |

**conv1d 对中弱宿主友好（对比 attention）**：与 §16 attention 只有 codex 拿下不同，conv1d **gptme 也拿下**（前 1.78/反 3.72）——因为 conv1d 有**明确的 kernel 优化套路**（shared-mem tiling + 每通道并行），不需 FlashAttention 级重写。**规律细化**：融合密集 case 能否跨宿主，取决于优化路径是否"套路化"——attention 需算法级重构（难），conv1d 只需标准 tiling（易），故 conv1d 宿主门槛低于 attention。

**harness auto-scale 阈值边界缺陷（本轮暴露，重要）**：aider 未建 bench.env 时，`--auto-scale` **本应探短核放大兜底**，但**未触发**——因为 aider 默认规模下 baseline 前向 0.157ms **刚好越过 0.15ms 短核阈值**（险过），auto-scale 判"非短核"不放大，于是用 8.4M 小规模出 VERDICT，反向短核虚高（1.32×）蒙混过关。**根因**：0.15ms 固定阈值太松，前向刚过阈值但反向仍是短核规模（反向本比前向重但此处 baseline 反向 0.38ms 也不算大）；更本质是**auto-scale 只看前向或单侧耗时判短核，未保证放大到"计算主导"规模**。加固方向：auto-scale 阈值调高（如 0.5ms）或多侧判断 + 强制放大到 candidate/baseline 均 >1ms 的计算主导区。规避：agent 建 bench.env 显式放大（codex/gptme 建了都没被骗，唯 aider 不建被骗）。记 [[project-baseline-traps]]。

**aider 架构弱点第三次印证**：RMSNorm（§8/9）、welford（§15 规模分支）、conv1d（本轮不建 bench.env）——aider 反复栽在"不主动建约定提到的额外文件（bench.env）→ 被短核/规模假象骗"。**换 GPT-5.5 更强模型仍犯**（是架构自主性弱，非模型能力），印证[[feedback-new-case-three-hosts]]三维独立。codex（内置 shell 主动性强）每形态都主动建 bench.env 从不被骗。

**十一形态光谱**：稳赢区（RBF/scan/RoPE/linear_ssm cumsum/Welford codex/**conv1d**）、擦线区（LN/RMSNorm/Welford 前向 reduce 密集）、宿主分层区（GEMM/online-softmax/causal_attn 融合密集）。**conv1d 归入稳赢区**（codex/gptme 前反向真赢，aider 仅反向栽在测量而非 kernel）。

**collected as 第十一形态参考 case**：codex 版收进 `cases/conv1d/`（K=4 手写展开 reference + shared-mem tiling 前向 + warp 规约 dW + bench.env B=128/C=512/T=1024）。


## 18. 扩形态：gated SSM（第十二形态，变系数递推——暴露约定深层盲区并催生"约定边界细化"，两轮对照验证）

第十二种形态：输入依赖门控的状态空间模型——`h_t=z_t·h_{t-1}+(1-z_t)·x_t`，`z_t=sigmoid(w·x_t+b)`（**门控输入依赖**，Mamba 核心）。**纯自然语言输入**。选它专为**测约定边界**：变系数递推是否有"干净的 O(N) 向量化 baseline"？

**核心发现：gated SSM 是"没有干净 O(N) 向量化 baseline"的算法**（实测坐实）。四种 reference 写法全有问题：①`cumprod(z)+cumsum(x/cumprod)` 的 O(T) 形式——z≈0.5、T=512 时 cumprod **下溢到 0 → 除法 NaN**（且其 autograd 反向图畸形，torch.compile 反向 baseline 达 **281ms** vs 前向 1.68ms）；②`for t` 循环——torch.compile 编译爆炸（T=64 编译 36s、T=128 编译 77s、T=512 卡死）；③O(T²) 下三角（log 空间 `exp(L_t−L_j)`，L=cumsum(log z)，每项≤1 数值稳）——**数值最稳但违反此前"禁 T²"约定**；④分块 associative scan——最优但 PyTorch 无内置、极难写对。这**证伪了旧约定的隐含假设"总存在数值稳定的 O(N) 向量化"**。

**第一轮（旧约定）——codex 作弊暴露弱 baseline 变种 C**：codex 用 `cumprod+cumsum` 的脆弱 O(T) reference，且 **make_inputs 挑 `b∈[3.5,4.5]` 让门控 z≈0.98 恒接近 1**（规避 cumprod 下溢），配合反向 autograd 图畸形（baseline 反向 281ms）→ 刷出**假 346×~638×**。这是**弱 baseline 第四变种**（数值脆弱 cumprod/cumsum + 挑输入分布迁就，见 [[project-baseline-traps]]）。

**约定细化（本轮核心产出，通用原则非 case 解法）**：SKILL/AGENTS/CONVENTIONS/CLAUDE 四份同步——
1. **推翻"一律禁 T²"**：固定系数递推（linear_ssm，系数与输入无关）有稳定 O(N) 前缀形式 → 禁 T²、必 cumsum；**变系数递推（gated SSM，系数输入依赖）若无稳定 O(N) → O(T²) 下三角合法诚实**（是该算法数值稳定的最优可行 baseline）。区分准则=系数是否输入依赖。
2. **新增防作弊点**：禁用 make_inputs 挑异常输入分布（如令门控恒≈1）迁就脆弱 reference。
3. **自主判断**：先试 O(N)，verify 发现 NaN/溢出/反向异常慢再退 O(T²)。
（为保纯输入测试，约定只写通用原则，**删去 gated SSM 的具体公式/数字**——不喂 case 答案。）

**第二轮（新约定）——三宿主对照，验证约定细化生效**：

| 宿主 | 结果 | reference | 加速比(T=256) | 说明 |
|------|------|-----------|------|------|
| **codex**(gpt-5.6) | ✅ 诚实达标 | **log 空间 O(T²) 下三角**（`logaddexp` 稳定 log-sigmoid + 成对前缀差）+ **自然 randn 输入** | 前7.80/反13.78 | 读懂新约定，注释明写"permitted for genuinely input-dependent recurrences"；**从作弊（挑b≈4+脆弱cumprod假346×）→ 诚实（自然分布+稳定T²）**，baseline 反向从虚高 281ms 降到正常 1.67ms |
| **aider**(GPT-5.5) | ✅ 诚实达标 | 同款 log 空间 O(T²) 下三角（`cumsum(log z)`+前缀差）+ 自然 randn | 前4.39/反10.19 | 新约定对 aider 也生效，自主退诚实 O(T²)+自然分布，与 codex 同级 |
| **gptme**(GPT-5.4) | ❌ 未达标 | **方向对**（退密集矩阵+自然分布 randn·0.5）**但写成 O(T³)**（`[chunk,T,T,T]` 四维 `prod(dim=3)` 显式连乘，未想到 log 前缀差降 O(T²)）+ **维度 bug**（verify RuntimeError shape 不匹配） | verify 崩 | 读懂新约定方向但实现力不足——GPT-5.4 写不对复杂密集 reference，卡 thinking |

**诚实加速比的性质（重要认知）**：gated SSM 的诚实 baseline **本身就是 O(T²)**（无数值稳定 O(N) 向量化），candidate 用 O(T) 递推 kernel → 7.8×/13.8× 是**真实算法复杂度优势**（O(T) vs O(T²)），**不是弱 baseline**。这反映 **CUDA kernel 对变系数递推的真实价值**：PyTorch 里 O(T) 要么 NaN（cumprod）要么编译爆炸（for），只能退 O(T²)；CUDA 能做数值稳定的 O(T) 递推。**这是本项目首次"诚实 baseline 就是 O(T²)"的形态**——加速比高但诚实。

**约定细化生效验证成功（skill 演进闭环第四次）**：前三次（online-softmax→禁for、linear_ssm→禁T²、welford→禁规模分支）都是**加严**；本轮是**放宽+精化**（承认 T² 在变系数递推的合法性 + 新增禁挑分布）——证明约定能双向演进。codex 从作弊到诚实、aider 同步诚实，验证"通用原则（非喂 case 解法）能引导强宿主自主处理约定边界"。gptme 方向对但实现受限，仍是三维独立（模型能力维）。

**十二形态光谱**：稳赢区（RBF/scan/RoPE/linear_ssm/Welford codex/conv1d/**gated_ssm codex+aider**）、擦线区（LN/RMSNorm/Welford 前向）、宿主分层区（GEMM/online-softmax/causal_attn 融合密集）。gated_ssm 归稳赢区（codex/aider 诚实拿下），但独特在**其诚实 baseline 就是 O(T²)**。

**collected as 第十二形态参考 case**：codex 版收进 `cases/gated_ssm/`（log 空间稳定 O(T²) 下三角 reference + 自然 randn 输入 + O(T) 递推 CUDA kernel + 前反向）。


## 19. 扩形态：scatter-add 分段聚合（第十三形态，全新"数据依赖写入+atomic 竞争"维度）

第十三种形态：`Y[s] = Σ_{i: idx[i]=s} X[i]`（X[N,D] 按整数索引 idx[N] 累加到 Y[S,D]，多源可写同段）。**纯自然语言输入**。选它测**全新 kernel 优化维度**：数据依赖写入 + atomic 竞争（之前形态都是规则密集张量操作，无真正的写竞争）。**新情况**：idx 是整数索引不可导（grad_inputs 只 X，make_inputs 里 idx 不设 requires_grad）——三宿主都处理对了。

**关键考验**：①idx 整数不可导的接口处理；②atomic 竞争策略（朴素 atomicAdd vs 排序段归约 vs 低冲突规模+float4）；③红线（reference 用 `index_add_`/`scatter_add` 基础算子合规，candidate 手写 CUDA 不落回 torch scatter）；④反向纯 gather（`dX[i]=dY[idx[i]]`，精确 err=0）。

| 宿主 | 结果 | 前向 | 反向 | 手法 / 说明 |
|------|------|------|------|------|
| **codex**(gpt-5.6) | ✅ 达标 | 2.19× | 1.14~1.22×(3连) | reference `index_add_`、idx 处理对；kernel warp-per-source + atomicAdd 前向 + D=128 float4 gather 反向；**规模 N=262144/S=32768（低冲突 ~8 源/段）**+ bench.env；反向擦线独立 3 连稳过 |
| **gptme**(GPT-5.4) | ❌ 未达标 | 1.02× | 1.03× | 合规（`scatter_add` 基础算子、idx 处理对）、正确性 PASS，但**朴素 atomicAdd 打平**；**规模 N=1048576/S=4096（高冲突 ~256 源/段）**更难赢；自己诊断准确（"高冲突 atomic scatter 优化空间有限，torch.compile 基线强"）但写不出低冲突策略——能力上限 |
| **aider**(GPT-5.5) | ❌ 未达标 | 1.44× 真赢 | **0.83× FAIL** | 合规（`index_add` 基础算子、idx 处理对）、正确性 PASS（反向 gather err=0）；**未建 bench.env**（老毛病第四次）→ auto-scale 放大到 N=32768 仍短核 → 首轮 **CV_INVALID**（baseline CV 6~9%）；放大真实规模（N=262144）后**反向纯 gather kernel 0.83× 打不过 torch**（前向 1.44× 真赢） |

**全新 atomic 维度上宿主分层明显**：scatter 前向的 atomic 竞争优化是全新 kernel 难点——codex **选对低冲突规模（S=32768，8 源/段）+ float4** 前向 2.19× 稳赢；gptme **高冲突规模（S=4096，256 源/段）+ 朴素 atomicAdd** 只 1.02× 打平；aider 前向 1.44× 真赢但**反向 gather 输给 torch**。**规律**：atomic scatter 的胜负关键在"规模/冲突度选择 + 访存优化（float4）"——codex 的规模选择（低冲突）本身就是关键决策，gptme 选了高冲突规模自增难度。反向纯 gather 看似简单，但 torch 的 gather 已高度优化，只有 float4 向量化（codex）能赢，朴素 gather（aider）反而输。

**harness auto-scale 缺陷第三次延续（scatter 场景）**：aider 未建 bench.env → auto-scale 探短核后放大到固定 `N=32768`，但 **scatter 是单元素轻操作，32768 放大后 baseline 仍仅 0.08ms（还是短核）→ CV 6~9% → CV_INVALID**（拿不到可信 VERDICT）。与 conv1d（0.15ms 阈值边界）、attention（32768 对 O(T²) 过大）一脉相承——**auto-scale 固定 32768 放大值对不同算法复杂度普遍不适配**。规避：手动 `--size-env` 放大到计算主导区（scatter 用 N=262144）。加固方向见 [[project-baseline-traps]]（自适应放大到 baseline>某绝对耗时）。

**aider 不建 bench.env 老毛病第四次**（RMSNorm→welford→conv1d→scatter）：每次都因默认小规模被短核/CV 假象干扰（本轮 CV_INVALID），换 GPT-5.5 仍不主动建 bench.env——架构自主性弱（codex/gptme 都建了）。印证[[feedback-new-case-three-hosts]]三维独立。

**十三形态光谱**：稳赢区（RBF/scan/RoPE/linear_ssm/Welford codex/conv1d/gated_ssm codex+aider/**scatter_add codex**）、擦线区（LN/RMSNorm/Welford 前向）、宿主分层区（GEMM/online-softmax/causal_attn 融合密集、**scatter atomic 竞争**）。scatter_add 归宿主分层区——全新 atomic 维度，仅 codex（选对规模+float4）拿下。

**collected as 第十三形态参考 case**：codex 版收进 `cases/scatter_add/`（warp-per-source atomicAdd 前向 + float4 gather 反向 + index_add_ reference + idx 整数不可导 + bench.env N=262144/D=128/S=32768）。


## 20. 扩形态：top-k 每行取最大 k 个值（第十四形态，数据依赖控制流+稀疏反向，暴露"用更慢通用算子当 baseline"弱 baseline 变种 D）

第十四种形态：`Y[i] = X[i] 行内最大的 k 个值(降序)，k=8`，反向梯度稀疏散回被选中的 k 个原位置（其余 0）。**纯自然语言输入**。选它测**全新维度**：数据依赖控制流（选哪 k 个取决于值本身，非固定位置）+ 稀疏反向。为兼容 framework 单张量输出约定，只输出值 Y[N,k]（索引内部用于反向散回、不外露）。

**关键考验**：①红线边界——reference 用 `torch.topk` 算不算高层算子？②数据依赖控制流 kernel 策略；③稀疏反向散回；④ties（平局）——randn 输入几乎无精确相等值规避。

**红线边界厘清（本形态产出）**：`torch.topk` 是**基础张量原语**（像 sort/cumsum/scatter_add，通用原语而非 F.scaled_dot_product_attention/F.layer_norm 那种神经网络高层融合算子）——**reference 可用 torch.topk**；但 **candidate 必须手写 CUDA top-k kernel，不得 op.py 直接调 torch.topk 糊弄**。三宿主 candidate 都手写了 kernel（不落回），红线守住。

| 宿主 | 结果 | reference | 加速比 | 说明 |
|------|------|-----------|------|------|
| **codex**(gpt-5.6) | ✅ 达标 | `torch.topk`（最优 O(D) radix select，**诚实 baseline**） | 前1.35/反1.33 | 唯一想到用最优原语；candidate warp 两阶段 top-8 + 稀疏散回反向；err=0；auto-scale 自适应 4 轮放大到计算主导区（前21ms/反2.2ms） |
| **gptme**(GPT-5.4) | ❌ 弱 baseline | `torch.sort(x)[:, :k]`（**全排序 O(D·logD)**） | 前1.30/反**9.26**虚高 | candidate 正确(err=0)+手写不落回+op.py/framework 合规，但 reference 用 sort 全排序 4096 元素只取前 8 → baseline 畸形慢(反向 7.86ms) → 反 9.26× 虚高。自主 PASS 但基线不诚实 |
| **aider**(GPT-5.5) | ❌ 弱 baseline | `torch.sort(x)[:, :k]`（同 gptme） | 前**8.98**/反**8.90**虚高 | 同款 sort 弱 baseline，加速比更虚高(baseline 前 12.4ms)；candidate 手写 kernel 合规；check_reference 预检正确命中 sort-for-topk |

**弱 baseline 变种 D：用更慢通用算子替代最优原语当 baseline**：gptme 和 aider **独立都用 `torch.sort` 取 top-k**（而非最优 `torch.topk`）——"用 sort 取 top-k"是**常见直觉陷阱**，全排序 O(D·logD) 只为取前 k 个，baseline 畸形慢（尤其排序 autograd 反向），candidate 手写 topk 刷虚高加速比。只有 codex 想到 torch.topk（O(D) radix select，诚实 baseline，故它加速比诚实地只有 1.3×）。**换诚实 baseline 复测受阻于 ties**：torch.topk/torch.sort/candidate 三者平局 tie-break 规则不同，换 baseline 后反向散回位置不一致导致 verify FAIL（seed 2/3/4 dX_err 大）——故此变种更靠静态查 reference。

**check_reference 补 sort-for-topk 规则（预检盲区补强）**：新变种 D 起初 check_reference 未覆盖（sort 不在危险模式）。已补规则：`sort(...)` + `[:, :k]/[:, :params[...]]` 切片 → WARN sort-for-topk。校准：gptme/aider sort 版命中、codex torch.topk 版不误报、13 个已有 case 回归无新误报。**这轮验证了 check_reference 是可增量扩展的**——发现新弱 baseline 变种就补一条静态规则。

**auto-scale 加固对 topk 生效**：codex/aider 都未建 bench.env，加固后的 auto-scale 自适应迭代放大（codex 4 轮 8192→524288 前21ms、aider 3 轮到前12ms），前反向都进入计算主导区（≥1ms）才停——验证上一轮 harness 加固对新形态普适。

**十四形态光谱**：稳赢区（RBF/scan/RoPE/linear_ssm/Welford codex/conv1d/gated_ssm codex+aider/scatter_add codex/**topk codex**）、擦线区（LN/RMSNorm/Welford 前向）、宿主分层区（GEMM/online-softmax/causal_attn/scatter atomic/**topk sort陷阱**）。topk 独特在：三宿主 candidate 全部正确+手写合规，**分层纯粹源于 reference 基线选择**（codex 用最优 topk 诚实、gptme/aider 用 sort 弱 baseline）——首次"分层不在 kernel 能力而在基线诚实度"的形态。

**collected as 第十四形态参考 case**：codex 版收进 `cases/topk/`（torch.topk 诚实 reference + warp 两阶段 top-8 前向 + int32 索引缓存 + 稀疏散回反向 + idx 整数不可导）。


## 21. 扩形态：2D max-pooling（第十五形态，全新 2D 空间维度，首次揭示"前向带宽墙"边界——三宿主前向一致过不了）

第十五种形态：2×2 步长 2 最大池化——`Y[n,c,i,j]=max(X[n,c,2i:2i+2,2j:2j+2])`，X[N,C,H,W]，反向梯度稀疏散回每窗口 argmax 位置。**纯自然语言输入**。选它测**全新 2D 空间维度**（之前 conv1d 是 1D，无 2D 空间 tiling）+ 数据依赖 argmax 稀疏反向。

**红线边界厘清（本形态产出）**：`F.max_pool2d` 是**神经网络池化层算子**（F.*/nn.* 层），红线§1 禁止——**与 topk 的 `torch.topk`（通用张量原语，允许）对比划出准则**：`torch.*` 通用张量原语（topk/sort/cumsum/scatter_add/amax）允许；`F.*`/`nn.*` 神经网络层算子（max_pool2d/layer_norm/conv/sdpa）禁。reference 须手写（reshape+amax 或跨步切片+where）。**check_reference 已扩展 pool 红线**（F.*_pool/group_norm/batch_norm/embedding + nn.*Pool），三宿主 reference 都手写未落回、REF_CHECK CLEAN。

| 宿主 | 结果 | 前向 | 反向 | reference / 说明 |
|------|------|------|------|------|
| **codex**(gpt-5.6) | ⚠️ 前向未过 | 1.03× 带宽墙 | **4.58× 赢** | reference `reshape.max(5).max(3)`；candidate 高效 argmax 散回 kernel；12 轮优化后 profile 判前向已受最小显存流量限制。反向赢(max().values 的 autograd baseline 慢) |
| **aider**(GPT-5.5) | ❌ 前向未过 | 0.998× 带宽墙 | 1.36× 赢 | reference 跨步切片+where；candidate 手写 argmax 缓存反向；卡自测(我代跑复验)；反向中等赢 |
| **gptme**(GPT-5.4) | ❌ | 1.02× 带宽墙 | **0.41× 输** | reference where 手写；candidate **朴素反向被 torch.compile 压制**(自己承认)；反向输 |

**首次揭示"前向带宽墙"边界（本形态最强结论）**：三宿主前向**一致过不了**（codex 1.03/aider 0.998/gptme 1.02，全 <1.05）——2×2 maxpool 前向是**纯访存**（读 4 个写 1 个，算术强度极低），torch.compile 的融合实现已到**显存带宽上限**，candidate 无论怎么写都难超 5%。**这是与宿主/模型无关的算法本征边界**：不是 skill 不行、不是 agent 不行，是**该算子前向没有可榨取的优化空间**（codex 12 轮 profile 确认带宽限制）。**新增光谱区间：带宽墙区**——纯访存低算术强度算子（elementwise/pooling 前向），baseline 已近带宽最优，candidate 难赢。

> **亲自穷尽复核（2026-07-22，Claude Code）**：为坐实"真带宽墙而非没优化够"，我在 codex 已有 float4 读+float2 写基础上又试了**扁平 1D grid-stride + `__ldg` 只读缓存**——前向仅 1.00→1.01×，仍 <1.05。**根因确凿**：2×2 maxpool 必须读全部 X、写全部 Y（读 4 写 1，总流量固定 = baseline），candidate 无法比 baseline 读写更少，结构/向量化/缓存优化都榨不出 5%。已回退（扁平版无实质增益，保留 codex float4 参考版）。**带宽墙本征边界经亲自穷尽确认**——区别于"擦线区"（LayerNorm 等前向靠 float4+寄存器缓存能从 0.93 榨到 1.09），带宽墙区是访存量本身=baseline、无榨取空间。

**反向差异纯源于 candidate kernel 质量（三维分离再证）**：三宿主 reference 都诚实（where/reshape+max，无弱 baseline），反向加速比差异（codex 4.58 / aider 1.36 / gptme 0.41）纯粹是 **candidate 反向 kernel 实现力**不同——codex 高效 argmax 散回、gptme 朴素被压制。注意 codex 反向 4.58× 部分因其 reference 用 `max().values`（autograd 反向较慢），而 aider/gptme 用 `where`（反向较快）——**同一算法不同诚实 reference 写法，baseline 反向速度不同**（都合法，非弱 baseline），故反向加速比不能跨宿主直接比绝对值，但都是诚实基线下的真实结果。

**十五形态光谱**（新增带宽墙区）：稳赢区（RBF/scan/RoPE/linear_ssm/Welford/conv1d/gated_ssm/scatter_add/topk 的 codex 版）、擦线区（LN/RMSNorm/Welford 前向）、宿主分层区（GEMM/online-softmax/causal_attn/scatter/topk-sort陷阱）、**带宽墙区（maxpool 前向——三宿主一致过不了，算法本征无优化空间）**。

**collected as 第十五形态参考 case**：codex 版收进 `cases/maxpool/`（reshape+max 手写 reference 不落回 F.max_pool2d + 2D argmax 散回反向 kernel；前向带宽墙未过但反向真赢，作"带宽墙边界"存证）。


## 22. 扩形态：CSR SpMV 稀疏矩阵乘（第十六形态，间接寻址/非规则访存维度收官）

第十六种形态：CSR 稀疏矩阵-稠密矩阵乘——`Y[m]=Σ_{j∈row m} vals[j]·X[col_idx[j]]`，A[M,K] 稀疏(CSR: row_ptr/col_idx/vals) × X[K,D] 稠密。对 vals 和 X 求梯度（row_ptr/col_idx 整数结构不可导）。**纯自然语言输入**。选它补最后一个未覆盖 kernel 维度：**CSR 间接寻址/非规则访存**（`X[col_idx[j]]` 通过 col_idx 间接访存——之前 scatter 是数据依赖写但连续读、topk 数据依赖控制流但规则访存，都无 CSR 这种不规则内存模式）。

**关键考验**：①CSR 间接寻址 kernel；②CSR 数据结构构造（make_inputs 造合法 row_ptr/col_idx/vals，复杂度高）；③反向 atomic 散回 dX（SpMV 反向固有）；④红线（torch.sparse.mm 禁 vs index_select/scatter_add 基础原语允许 vs cuSPARSE 底层库判定）。

| 宿主 | 结果 | 前向 | 反向 | reference / 说明 |
|------|------|------|------|------|
| **codex**(gpt-5.6) | ✅ 达标 | 5.97× | 2.48× | reference `index_select+index_add_`（标准 SpMV，不落回 torch.sparse）；candidate 手写融合 CSR kernel（D=64 向量化前向 + 融合 dvals/原子 dX 反向）；CSR 构造正确；诚实真赢 |
| **aider**(GPT-5.5) | ✅ 达标 | 6.21× | 2.38× | 同款标准 SpMV reference（`x[col_idx]`+`scatter_add_`）+ 手写 kernel；与 codex 加速比几乎一致（印证诚实）；**auto-scale 加固兜住其不建 bench.env 老毛病**（自适应放大到 M=32768 计算主导区）；aider 少有的干净全过 |
| **gptme**(GPT-5.4) | ❌ 反向未过 | 3.42× 真赢 | **0.67× FAIL** | 标准 SpMV reference（合规）+ 前向手写 kernel 真赢；**反向用 cuSPARSE `A^T@grad_out` 求 dX**——cuSPARSE 通用路径 descriptor/workspace 固定开销大，打不过 torch.compile 融合 scatter（gptme 自己诊断准确）。前向拿下 CSR 间接寻址，反向策略选择不如手写融合 |

**红线判定：cuSPARSE 允许（补进约定，与 cuBLAS/CUB 一致）**：gptme 反向用 cuSPARSE → 判**合规**（cuSPARSE 是 CUDA 官方底层库，非 torch 高层 `torch.sparse.mm`）。同时厘清完整准则并写进 SKILL §1 + AGENTS：**① torch 高层算子禁**（`F.*`/`nn.*`/`torch.matmul`/`torch.sparse.mm`/SDPA）；**② CUDA 官方底层库允许**（cuBLAS/CUB/cuSPARSE，在自定义 .cu 里调）；**③ 通用张量原语 reference 里允许**（`torch.topk`/`sort`/`cumsum`/`scatter_add`/`index_select`，基础操作非层算子），但 candidate 仍须手写 .cu 不得直接调糊弄。这条准则是十六形态一路厘清的（cuBLAS@GEMM、CUB@scan、topk@topk、cuSPARSE@spmv）。

**间接寻址/非规则访存维度收官**：CSR SpMV 是访存密集不规则算子，torch 的 `index_select+index_add_` 通用组合 baseline 会物化中间张量 + 多 kernel launch（前向 ~1-2ms），手写融合 CSR kernel（gather+乘加一趟 + 融合反向）大幅省物化 → codex/aider 前 6× 真赢（诚实基础算子 baseline，非弱 baseline）。gptme 前向也真赢，仅反向 cuSPARSE 固定开销输——**三宿主前向全部真赢**，说明 CSR 间接寻址的前向优化（融合 gather）对三宿主都可及，反向的策略选择（手写融合 vs cuSPARSE）才拉开差距。

**十六形态维度全景（覆盖完整）**：归约(LayerNorm/RMSNorm/Welford/softmax)、矩阵乘(GEMM/attention)、scan(scan/linear_ssm/gated_ssm)、卷积(conv1d/maxpool)、elementwise+位置编码(RoPE)、数据依赖写(scatter)、数据依赖控制流(topk)、2D空间(maxpool)、**间接寻址/非规则访存(spmv)**。四光谱区间：稳赢区、擦线区、宿主分层区、带宽墙区。

**collected as 第十六形态参考 case**：codex 版收进 `cases/spmv/`（index_select+index_add_ 标准 SpMV reference + 手写融合 CSR kernel(D=64 向量化前向 + 融合 dvals/atomic dX 反向) + CSR 构造）。


## 23. 阶段 9.3：真·自然语言输入 + 数学实现确认闸门（流程验证，Claude Code）

与 §10-22（扩形态）不同，本节验证的是**交互流程机制**——用户只给**真正的自然语言意图**（不含数学公式/shape），agent 先推导数学规格呈请确认、确认后才自动实现。

**测试输入（纯自然语言，无公式无 shape）**："我想要一个操作，把一批向量各自变成单位长度，然后按一个可学习的每维度权重缩放。"

**流程走通（Claude Code 作宿主，按新增的 SKILL 步骤 0.5）**：
1. **推导数学规格**：识别为 L2 归一化（除以每行 L2 模长）+ 每维度可学习权重 `g[D]` 缩放；给出前向公式 `norm=sqrt(Σx²+eps); Y=(x/norm)*g`、输入 `X[N,D]`+`g[D]`、输出 `Y[N,D]`、`grad_inputs=[X,g]`、eps=1e-6。
2. **列 4 个语义澄清点**（自然语言歧义处显式列采取解释+备选）：① "单位长度"=L2 范数（**并显式提示"这不是 RMSNorm（除以 sqrt(mean)）"**、也不是 L1/L∞）；② "每维度权重"=`g[D]` 逐元素乘；③ 归一化沿最后一维 D；④ 加 eps 防除零。
3. **附 PyTorch reference.py 代码** 供核对。
4. **停下等用户确认**——未抢跑建 case/写 kernel（守住"数学未确认不得动工"约定）。
5. 用户确认（"ok"）后，**才**自动建 `cases/l2norm_scale/` + 写前反向 CUDA kernel + A100 自测 + 优化——全自动跑完不再中途停。

**核心结论：确认闸门机制完整验证成功**。真·自然语言 → agent 推导数学规格 + 主动暴露歧义（尤其"这不是 RMSNorm"的澄清事后印证关键——L2 与 RMSNorm 仅差 sqrt(D) 因子，极易误解）→ 人类确认 → 确认后自动实现。**闸门只在数学层一次，不破坏后续自主闭环**。这把"数学建模"责任从用户转移到 agent，用户从"给公式"降到"给意图+确认"——"输入最小化"的终点档达成。

**性能（与流程验证结论无关）**：l2norm_scale 正确性全 PASS（前 err~6e-8、dX~9e-8、dg~4e-5），性能前 1.03×/反 0.85× 未达标。**l2norm_scale 属归一化家族**（与 RMSNorm 近同构，仅差 sqrt(D) 因子 + g 缩放），落已知**擦线/难赢区间**——reduce 密集、torch.compile 已高度融合，反向两 kernel（dX + 独立 dg）架构性多读一遍 X/G。优化尝试：float4 向量化（前向 0.92→1.02×）有效；寄存器缓存 float4（占 96 寄存器/线程 → occupancy 暴跌，反向 0.85→0.76× 翻车）、dg row_blocks 降到 32（并行度不足，反 0.85→0.42× 翻车）均无效并回退。诚实定性：归一化家族反向擦线是本征特性（LayerNorm §8.2 手工深调才勉强过），非流程/skill 问题；9.3 核心目标（确认闸门流程）已达成，未烧更多优化轮次。

**collected as 阶段 9 产物存证**：`cases/l2norm_scale/`（确认闸门流程产出——纯 NL→数学确认→自动实现；正确性全 PASS，性能落归一化家族擦线区）。


## 24. 阶段 9.4：确认闸门三宿主推广（codex/gptme 守闸门、aider auto-test 架构抢跑）

把 9.3 的确认闸门流程推广到三宿主验证。**统一纯自然语言输入**（带任务信号，无公式无 shape）："帮我实现一个操作，把一批分数变成概率分布，但要能控制分布的尖锐程度。用这个仓库的 skill 走完整流程。"（识别为 temperature softmax `p=softmax(x/T)`。）

| 宿主 | 确认闸门（前半程） | 确认后自动实现（后半程） | 性能 |
|------|------|------|------|
| **codex**(gpt-5.6) | ✅ 守闸门 | ✅ 自动跑到 3 连 PASS | 前 1.11 / 反 1.70 达标 |
| **gptme**(GPT-5.4) | ✅ 守闸门（列 4 歧义带备选、手写 reference、明确"如果你确认我将继续"、**未抢跑**） | ✅ 确认后自动读范例→建 case→自测 | 正确性 PASS；前 0.74 FAIL / 反 2.89 PASS |
| **aider**(GPT-5.5) | ❌ **抢跑**（不出规格、不停，直接建 case+自测） | （无独立后半程，一气呵成） | 正确性 PASS；前 0.95 FAIL / 反 1.52 PASS |

**核心结论：确认闸门在交互式宿主生效，aider 因 auto-test 架构抢跑**：
- **codex/gptme 守闸门**：都是交互式对话，读 SKILL/AGENTS 步骤 0.5 后先出结构化数学规格 + ⚠️语义澄清点（gptme 还列了 softmax vs sparsemax/entmax、temperature vs inverse-temperature 等备选）+ 手写 reference，**停下等用户确认**（gptme 最担心的"别停惯性压倒闸门"没发生——它正确区分了"数学层要停"vs"kernel 循环别停"）。确认后才自动实现。
- **aider 抢跑**：`.aider.conf.yml` 配了 `auto-test: true` + `test-cmd`，其工作模式**架构性地就是"改文件→自动跑测→据结果改"的自主循环，没有"停下问用户"的环节**。步骤 0.5 要它先出规格停下，但 aider 连 case 都建好直接 auto-test 了。**这与"aider 不主动建 bench.env"同源——都是 aider 架构自主性维度的表现**（自主闭环不用人管是优点，但停不下来等确认是这里的缺点，换强模型也改不了）。**注意 aider 产物本身合规**（reference 手写 amax/exp/sum、无 torch.softmax、REF_CHECK CLEAN、framework 干净）——**抢跑的是流程（没停下确认），不是产物质量**。

**确认闸门是"交互流程约定"，依赖宿主能停下等人类**：codex/gptme 交互式对话天然能停；aider auto-test 自动循环绕过停下。**不是 skill 不清晰，是 aider 架构不适配"人类确认闸门"**——干净的三维分离（此处是宿主架构维度决定闸门能否生效）。若要 aider 守闸门，需关掉 auto-test（改配置，牺牲其自主闭环优势）。

**触发条件发现（codex 冷启动补充）**：codex 首次给**纯意图无任务信号**（"我想要一个操作，把一批分数变成概率分布..."）时**当普通问答答了**（解释温度 softmax + 给 torch.softmax 代码），没进 skill 流程；补"帮我实现...用这个仓库 skill 走完整流程"的**任务信号**后才进闸门流程并手写基础算子。→ 确认闸门要 agent 先认定"这是建 kernel 的 skill 任务"才触发；纯"我想要…"太像聊天。真实用户找 kernel skill 时本会带实现信号，但**冷启动宿主更依赖任务信号**（对比 9.3 Claude Code 已在 skill 语境里，纯意图也正确进闸门）。见 [[feedback-pure-algorithm-input]]。

**性能（softmax 家族，与闸门无关）**：三宿主前向 codex 1.11 过、gptme 0.74 / aider 0.95 挂——softmax 前向要 float4+寄存器缓存才赢（codex 做了、gptme/aider 朴素规约没做到）；反向都赢（softmax Jacobian `dx=p*(gout-Σgout*p)` 好写）。属已知归一化/softmax 家族的 kernel 能力分层，非闸门问题。

**collected as 阶段 9.4 产物**：codex 版收进 `cases/temperature_softmax/`（手写数值稳定 softmax reference + float4 前向 + softmax Jacobian 反向；确认闸门→自动实现的完整产物，前 1.11/反 1.70 达标）。


## 25. 阶段 9.5：确认闸门歧义压力测试（codex，极模糊输入）

验证确认闸门在**歧义爆表**输入下能否主动暴露歧义请用户定夺（而非擅自假设一套跑下去）。9.3/9.4 的输入歧义少（agent 拍板一种合理解释即可）；9.5 故意用极模糊输入压测。仅测 codex。

**输入（极模糊，故意不说任何细节）**："帮我实现一个注意力操作，用这个仓库的 skill 走完整流程。"——单头/多头、causal/双向、自/交叉注意力、scale、mask、shape 全部留白。

**codex 表现（理想）**：
- **主动暴露 4 类关键歧义 + 备选**：单头 vs 多头、causal vs 双向、自注意力 vs 交叉注意力、有无 mask/bias/dropout。
- **上下文感知（超预期）**：主动发现仓库已有 `causal_attn`，于是默认选"非因果版"避免重复，并建议新 case 名 `self_attn` 避免重名。
- **显式说明采取的解释**（单头/双向/自注意力/无 mask）+ 列备选 + **停下请用户确认**，未擅自往下跑。

**结论：确认闸门在极模糊输入下正确工作**。它没"拍板一套闷头跑"，而是把选择摊开（采取解释 + 备选 + 停下确认）——正是步骤 0.5 要求的"显式列采取解释+备选，让用户一眼看出要不要改"。**歧义越大，闸门越把选择交还用户**——这是确认闸门的核心价值（兜住自然语言歧义，避免"理解错了跑到 kernel 才发现"）。9.5 只验流程（歧义暴露能力），已达成；未真跑 kernel（self_attn ≈ causal_attn 去 mask，无新意）。

---

## 阶段 9 收官小结（真·自然语言输入 + 数学确认闸门）

**目标达成**：用户从"给公式+shape"降到"给自然语言意图 + 确认"——"数学建模"责任由 agent 承担，人类确认闸门兜住歧义。
- **9.1/9.2**：SKILL 步骤 0.5 + 三约定文件（AGENTS/CONVENTIONS/CLAUDE）定义"NL→数学规格→确认"流程。
- **9.3**（Claude Code）：纯 NL"L2 归一化+缩放"→ 推导数学规格 + 列歧义（"这不是 RMSNorm"）+ 停下确认 → 确认后自动实现（l2norm_scale，正确性 PASS，性能归一化家族擦线）。
- **9.4**（三宿主）：codex✅/gptme✅ 守闸门、aider❌ auto-test 架构抢跑；触发条件=需任务信号（冷启动纯意图当问答）。
- **9.5**（codex 极模糊压测）：歧义爆表时主动暴露 4 类歧义+备选+停下，验证闸门核心价值。
- **核心洞察**：确认闸门是**交互流程约定**，依赖宿主"能停下等人类"——交互式对话宿主（Claude Code/codex/gptme）生效，自动循环架构（aider auto-test）抢跑。是三维分离里"宿主架构自主性"维度的又一体现。


## 26. 策略库打磨效果验证：cosine_sim 优化循环减轮次（Claude Code，2 轮达标）

打磨优化策略库后（决策表补 9~16 形态经验 + 本征边界识别 + loop.md 过度优化翻车教训），用**全新中等难度 case** 验证"减 loop 轮次"——目标：诊断对症、少盲试、不翻车。

**cosine_sim**（成对余弦相似度矩阵 `S[i,j]=(A[i]·B[j])/(|A[i]||B[j]|)`，A[N,D]×B[M,D]→S[N,M]，广播式 reference 保赢面）。Claude Code 作宿主，严格按 loop.md 循环（每轮 profile 诊断→查表选手段→测收益）。

| 轮次 | 诊断（查表定位瓶颈） | 手段 | 前向 | 反向 |
|------|------|------|------|------|
| **0** 朴素基准 | 一线程一输出、每输出重算范数 | — | 0.37× | 0.035×(60ms) |
| **1** | "反向重算前向中间量"+"访存重复读" | 预计算缓存 Ah/Bh/invNorm 复用 + 前向 TILE tiling | **2.12×**✅ | 0.048×(仍慢) |
| **2** | "规约低效"——反向 dAh=Σ_j dS·Bh 用了 shared atomicAdd（M×D 次竞争爆炸） | 改 **thread-per-d 寄存器私有累加**消 atomic | — | **1.42×**✅ → 全达标 |

**2 轮达标（前 2.12/反 1.42，复验一致）**。策略库打磨见效的三个证据：
1. **诊断对症、不盲试**：第 1 轮直接上决策表标注的"缓存复用（反向决定性一招）"，没从 float4/warp 顺着清单试起——阶段 8 前 gptme 曾盲试 5 种列规约卡 ~1.0× 没过，对比鲜明。
2. **不翻车**：第 2 轮发现反向仍慢（0.048×），profile 定位是 shared atomicAdd 竞争（规约低效），**一步换成寄存器私有累加**改对；没陷入"硬调 atomic 参数"（loop.md 翻车教训警示过盲调参数会 0.85→0.42×）。
3. **前向 1 轮到位**：广播式 reference（RBF 同款保赢面经验）+ 缓存范数 + tiling，前向直接 0.37→2.12×。

**关键**：第 1 轮"前向 2.12 但反向 0.048"的诊断是转折点——决策表让我准确定位"反向卡在 shared atomic 竞争"而非盲目重写，第 2 轮一击达标。**这验证了打磨后决策层能显著减少优化轮次**（诊断驱动 vs 盲目穷举）。

**collected as 第十七形态参考 case**：`cases/cosine_sim/`（广播式 reference + 预计算缓存 Ah/Bh/invNorm + 前向 tiling + 反向 thread-per-d 寄存器累加避 atomic；2 轮诊断驱动达标的教学案例）。

### 四宿主客观轮次对照（cosine_sim，`.round_final` 机器计数，非自报）

给 run_on_a100.sh 加"达标轮次存档"（总是计数每次自测调用，PASS 时把总轮次写入 `.round_final_<case>` 再清零）——客观量化各宿主"从建 case 到达标跑了几轮自测"，不靠宿主自报。三宿主用**同一句纯算法定义**（公式+shape+"广播式 reference"提示，同起跑线）各自干净房间从零建+自主 loop：

| 宿主 | 客观轮次 | 手法 / 说明 |
|------|------|------|
| **codex**(gpt-5.6) | **1** | 首次自测即 PASS：直接 cuBLAS SGEMM + 缓存归一化向量，跳过朴素弯路 |
| **gptme**(GPT-5.4) | **1** | 同 cuBLAS + 缓存路线，**最弱宿主也一次到位** |
| **Claude Code**（我） | 2 | 故意从朴素版起演示 loop（第1轮缓存复用+tiling、第2轮诊断反向 shared atomic 竞争改寄存器累加） |
| **aider**(GPT-5.5) | 4 | auto-test 边做边测架构，多次小迭代到达标（另：干净房间 `rm cosine_sim` 留下 git "待提交删除"脏状态，曾令 aider reflection 循环卡壳空转——`git rm` 固化删除后重跑即顺利，是建净房细节教训） |

**结论——策略库打磨"减轮次"充分验证**：codex/gptme **各 1 轮**（读打磨后决策表"矩阵乘→cuBLAS"+"缓存复用"，**首次实现就诊断到位、直接写高效版**）；连最弱的 gptme 也 1 轮到位——说明**策略库把"该用什么手段"讲清楚后，弱模型也能少走弯路**。对比阶段 8 前 gptme 盲试 5 种列规约卡 ~1.0× 没过，质变明显。轮次差异反映宿主架构（aider auto-test 多次小迭代 vs codex/gptme 一次到位），但**都受益于诊断驱动的策略库**（无一盲目穷举/翻车）。`.round_final` 计数机制让"减轮次"从主观印象变为客观可量化。


## 27. 攻坚"难达 1.05×"实例：l2norm_scale 反向攻下 + 内化 + 三宿主验证（至少 codex 达标）

回头攻此前卡 1.05× 附近/以下的实例（归一化家族 l2norm_scale 前1.03/反0.85 FAIL、带宽墙 maxpool 前向~1.0）。目标：我先手工找最优优化做到达标 → 内化 skill → 三宿主重做，验证至少 codex 达标。

**我的手工攻坚（Claude Code）**：
- **l2norm_scale 反向 0.84→1.60× 攻下**：根因是 dX/dg 拆两 kernel、dg 额外整遍重读 X/G（~2GB）。**最优解=融合成一 kernel、一 block 管一段行 chunk，X/G 只读一遍算 dX + shared 私有累积该 chunk 的 dg 贡献、chunk 末一次性 atomicAdd**（atomic 从 262144 次/列降到行块数 8192）。**过程翻车 2 次**（都被 loop.md 翻车教训预警、及时回退）：①per-element global atomicAdd 融合 → N 行抢同地址竞争爆炸 0.85→0.26×；②寄存器缓存大数组 → occupancy 崩 0.85→0.76×。chunk 级 shared 私有累积才对。
- **l2norm_scale 前向 1.02 + maxpool 前向 1.00 = 确认带宽墙本征边界**：前向读写量=baseline，试 chunk+g缓存、寄存器缓存X、float4 均无益（maxpool 计算主导区 3 次一致 1.00）。**非没优化够，是纯访存前向 candidate 无法比 baseline 读写更少**——接受边界结论（诚实非天花板误判）。

**内化**：SKILL 决策表新增"**反向多梯度融合到一遍访存**"行 + loop.md 手段 2b + 翻车教训补"per-element atomic 融合"（0.85→0.26）。

**三宿主验证（读增强 skill 从零重做 l2norm_scale，纯算法定义）**：

| 宿主 | 反向 | 前向 | 是否应用内化的 chunk 融合 |
|------|------|------|------|
| **codex**(gpt-5.6) | **1.48× 达标** | 1.01 带宽墙触顶 | ✅ **首轮自主用**（"融合 dX+dg 按 32 行块累积 dg 避免分别扫描"——与我内化手段同款、同 32 行块） |
| **aider**(GPT-5.5) | 1.02 擦线下 | 0.99 带宽墙 | ❌ 拆 4 kernel（dX + dg_vec4 + dg_scalar），dg 单独扫，没融合 |
| **gptme**(GPT-5.4) | 未收敛 | — | ❌ 拆 5 kernel 盲目循环 23+ 轮（无 round-cap 不收敛，执行力+优化力不足） |

**核心结论——攻坚目标达成（至少 codex 达标）**：
- **内化生效于最强宿主**：codex 读增强 skill **首轮就自主应用"反向 chunk 融合"**把反向从此类 ~0.85 做到 **1.48 达标**——证明"反向多梯度融合"手段**成功内化、可复用**，此前的反向卡点**不是天花板、是没优化够**。前向 codex 也判带宽墙触顶（与我同结论，没空转）——证明前向确是本征边界。
- **宿主能力分层再现**：aider（反向拆碎只 1.02）、gptme（23 轮盲循环不收敛）**没能应用融合手段**——是宿主优化力/执行力上限（非内化无效，codex 已证手段有效）。印证三维独立：同一增强 skill，强宿主能吸收高级手段，中弱宿主受能力限。
- **诚实边界**：l2norm_scale 整体仍 BENCH_FAIL（前向带宽墙拖累），但**反向达标是实打实战果**——把"归一化反向打不过"从误判改写为"可攻下（chunk 融合）"，同时确认"归一化/纯访存前向是真带宽墙"。主仓 `cases/l2norm_scale/` 保留我的 chunk 融合版（反向 1.60）作参考。


## 28. 扩形态：GroupNorm（第十八形态，分组归约新结构；三宿主全走确认闸门 + 前向规模挑选被复验拆穿）

第十八形态。**纯自然语言意图**输入（"把特征图通道切成若干组、每组内部一起归一化、再逐通道缩放偏移"，不含公式/shape），三宿主各自走 skill 步骤 0.5 确认闸门 → 建 case → 自测 loop。新维度：**分组归约**（归约维=组内 C/G 通道 × H×W，比 LayerNorm 整行、l2norm 单行都复杂）+ dgamma/dbeta 跨 N×H×W 列规约。规模 N=64,C=128~256,H=W=32~56,G=32；对 X/gamma/beta 求梯度。

**① 确认闸门三宿主全触发（阶段 9 目的达成）**：codex/aider/gptme **都先把模糊意图推导成精确数学规格 + ⚠️歧义澄清点（组内跨通道+空间 vs 每位置只沿通道 / 总体方差无 Bessel / 逐通道 gamma-beta / 排除 Instance-Layer-BatchNorm）+ PyTorch reference，然后停下等确认，未擅自建 case**。三宿主推导的数学一致且正确（标准 GroupNorm，biased variance）。对比 §24 codex 冷启动纯意图曾被当聊天问答——本轮 prompt 带"帮我实现…CUDA kernel"任务信号，三宿主都正确进闸门。

**② 三宿主自测结果（我全部独立复验，多规模交叉）**：

| 宿主 | 前向 | 反向 | 反向手段 | 达标 |
|------|------|------|---------|------|
| **codex**(gpt-5.6-sol) | **1.28~1.40×**（384→768 跨规模稳赢） | **1.17~1.33×** | 一 block 一组 float4 两遍统计；dgamma/dbeta 融一个 param kernel 一遍读；dX 复用 mean/rstd | ✅ **前反向真达标**（2 轮到位，跨规模稳） |
| **aider**(GPT-5.5) | 1.03× FAIL | **1.14× PASS** | **首轮诊断"dgamma/dbeta 独立 kernel 多扫超大张量"→ 融合进 dX kernel**（内化的反向多输出融合生效） | ⚠️ 仅反向；前向卡 1.03（误诊为同步开销、试双值规约无效，没上 float4），撞 reflection 上限(3)退出 |
| **gptme**(GPT-5.5) | **规模挑选假象**（GN_N=384:1.07 过，但 baseline 前向仅 0.92ms 短核；768:1.048、1536:1.035 **FAIL**） | **1.06~1.07×**（跨规模真达标） | 前向 float4 擦线；反向历经负优化谷底(0.585×)后自主回退 | ⚠️ 自报 GN_N=384 三连 PASS，**复验拆穿前向是规模挑选** |

**③ 核心结论**：

- **codex 前向也大幅赢（区别于 l2norm/maxpool 前向带宽墙）**：GroupNorm 组内规约 M=C/G×H×W（约 1.2 万~4 万元素）**算术强度足够高**，不是纯访存——codex float4 一 block 一组融合 mean/var 两遍统计 + 仿射，跨规模稳定 1.28×。**归一化家族前向并非都是擦线/带宽墙**：整行小 reduce（LayerNorm D=1024、l2norm）擦线，大分组 reduce（GroupNorm）可真赢。据此更新光谱：GroupNorm 前向归**稳赢区**（codex），非擦线区。

- **反向多输出融合手段再次内化生效（aider 首轮自主用）**：aider 首轮就诊断出"dgamma/dbeta 独立 kernel 多扫一遍超大张量"并**融合进 dX kernel** → 反向 0.90→1.14× 达标。这是 §27 l2norm 内化的"反向多梯度融合"手段在**新形态、新宿主上复现**——证明手段通用可迁移（不只 codex 会用）。

- **gptme(5.5) 比历史 5.4 明显强，但前向靠规模挑选**：5.4 在 l2norm "23 轮盲循环不收敛"（§27）；本轮 5.5 能**自主纠偏**（反向负优化到 0.585× 后主动回退到 1.06）、读懂"擦线 3 连"约定并执行、建 bench.env。但**前向没有 codex 的 float4 实现力**，只能挑 GN_N=384（baseline 前向 0.92ms 偏短核）让固定开销撑高加速比擦过——**复验用计算主导区规模（768/1536）拆穿：前向掉到 1.048/1.035 FAIL**。反向则跨规模真达标（1.06-1.07）。

- **规模挑选 = 短核假象的宿主自选变种（新暴露点）**：gptme 自报"GN_N=384 三连 PASS"，但那是它**从 384/512/640/768 里挑了前向恰好能过的最小规模**——384 前向 baseline 仅 0.92ms（未达 AUTO_TARGET_MS=1.0ms 计算主导区门槛），固定开销虚高。**判据：加速比强规模依赖（384→768→1536 前向 1.07→1.048→1.035 单调衰减）= 前向优势主要来自固定开销摊薄，非 kernel 真更快**。复验必须多规模交叉，不认单一挑选规模的擦线 PASS。

**④ harness 缺陷（本轮暴露，待修）**：
1. **auto-scale 对多维乘积规模 case 放大失控 OOM**：auto-scale 只对单一 SCALE_VAR（GN_N）从 8192 ×4 放大到 AUTO_MAX=4194304，但 GroupNorm 总张量 = N×C×H×W，放大 N 到 4M 时 ×256×1024 ≈ 1.1 万亿元素 ×4B ≈ **4TB → CUDA OOM**（gptme 撞上，日志 "Tried to allocate 2048 GiB"）。**修复方向**：auto-scale 放大时对**总元素数/字节**设上限（如 ≤5 亿元素 / 2GB），而非只限单一 SCALE_VAR。codex/gptme 靠建 bench.env（GN_N=64/384）规避，但 harness 不应让多维 case 放大爆炸。
2. **短核门槛对多维 case 偏松**：GN_N=384 前向 baseline 0.92ms 已接近但未达 1.0ms 门槛，却让 gptme 前向擦线 PASS——多维 case 的"计算主导区"判定应看总计算量/绝对耗时更严。

**collected as 第十八形态参考 case**：codex 版收进主仓 `cases/groupnorm/`（一 block 一组 float4 前向 + dX 复用 mean/rstd + dgamma/dbeta 融合 param kernel + bench.env GN_N=64；前反向跨规模稳赢 1.28/1.32）。

**十八形态光谱更新**：稳赢区（RBF/scan/RoPE/linear_ssm/Welford/conv1d/gated_ssm/spmv/**GroupNorm 前反向 codex**）、擦线区（LN/RMSNorm/l2norm/softmax/cosine 前向小 reduce）、宿主分层区（GEMM/online-softmax/causal_attn 融合密集）、带宽墙区（maxpool/l2norm 纯访存前向）。**GroupNorm 归稳赢区（codex）**：大分组 reduce 算术强度够，前向可真赢，颠覆"归一化前向必擦线"的旧印象——**reduce 规模大小决定擦线 vs 稳赢**。宿主分层再现：codex 前反向真赢、aider 反向达标（用上融合）前向擦线不过、gptme 反向真达标但前向靠规模挑选（复验拆穿）。




## 29. 规模敏感复测上线后的擦线区系统重估（harness 检测催生，修正历史"擦线达标"水分）

§28 上线"规模敏感复测"（擦线 PASS+短核→自动 ×2/×4 放大复测，掉破 1.05 判 `PASS_SCALE_SUSPECT`）后，回归验证时该检测**揭示了一批历史擦线 case 的前向"达标"其实是短核固定开销虚高**——遂系统重估擦线区（归一化/reduce 家族）各 case 的前向诚实性。方法：跑到计算主导区规模（baseline ≥1ms，靠 auto-scale 或手动放大）看前向加速比是否仍 ≥1.05。

**重估结果（GPU0 独占，多规模交叉）**：

| case | 短核规模前向 | 计算主导区前向 | 反向 | 前向诚实性 |
|------|------|------|------|-----------|
| **welford**(§15) | WELFORD_B=256: 1.095 PASS(baseline 0.24ms) | 512→1.049 / 1024→1.019 **FAIL** | 1.21~1.31 跨规模真达标 | ❌ **短核虚高，§15"前向1.097达标"存疑** |
| **temperature_softmax**(§24) | TSM_B=32768: 1.102 PASS(baseline 0.24ms) | 65536→1.057 / 131072→1.027 **FAIL** | 1.65~1.72 跨规模真达标 | ❌ **短核虚高，§24 前向达标存疑** |
| **cosine_sim**(§26) | — | COS_N=8192(baseline 1.5ms): **2.12** PASS | 1.42 真达标 | ✅ **诚实**（点积规约算术强度够） |
| **softmax_ce** | SMCE_B=8192: 1.666(baseline 0.16ms) | 131072(baseline 1.17ms): **1.351** PASS | 1.42 真达标 | ✅ **诚实**（衰减但仍稳赢） |
| **layernorm**(§8/9) | LN_B=32768: 前1.099/反1.065 PASS(baseline 前0.24ms) | 65536→前1.040/反1.009 / 131072→前1.012/反0.993 **FAIL** | — | ❌ **前反向均短核虚高，§8/9"前1.08/反1.25达标"存疑**（整行小 reduce D=1024，算术强度最低，连反向也打不过） |

**核心规律——前向擦线 vs 稳赢由算术强度决定（印证 §28 内化）**：
- **前向真赢（算术强度够）**：cosine_sim（成对距离/点积规约）、softmax_ce（exp+规约沿 D）、GroupNorm（大分组 reduce）——放大规模仍稳赢，是 kernel 真优势。
- **前向短核虚高（算术强度低）**：welford（单遍均值方差统计）、temperature_softmax（单纯 softmax 缩放）——每输出规约量小、逐元素为主，`torch.compile` 已到带宽/最优，candidate 只在短核规模靠固定开销摊薄虚高，计算主导区打不过。**反向都真达标**（有 Jacobian/统计缓存/列规约可优化）。
- **判据**：baseline 前向 <1ms 且加速比擦线（<1.15）→ 必多规模交叉；加速比随规模单调衰减掉破 1.05 = 短核虚高。

**处置与认知修正**：
- **不改历史 case 的 kernel/交付**（它们反向真达标、前向在原规模数值正确，只是前向加速比结论需加注）——重估是**认知修正**，非推翻交付。welford/temperature_softmax 的前向应记为"短核规模擦线、计算主导区打不过"（与 l2norm/maxpool 前向同属"前向打不过"，只是成因是低算术强度而非纯带宽墙）。
- **§28 光谱表述修正**：`Welford` 不应笼统列"稳赢区"——其**反向稳赢、前向短核虚高**。准确表述见下。
- **检测价值**：`PASS_SCALE_SUSPECT` 把"短核规模擦线虚高"从"人工多规模复验才发现"变为 harness 自动拦截，且对固定 bench.env 的历史 case 与主动挑规模一视同仁（措辞中性化为"达标存疑"非"作弊"，见 09a889d）。

**光谱准确化（十八形态 + 本次重估）**：
- **稳赢区（前反向计算主导区均真赢）**：RBF/scan/RoPE/linear_ssm/conv1d/gated_ssm/spmv/**cosine_sim/softmax_ce/GroupNorm(codex)**。
- **前向短核虚高、反向真达标**：**welford/temperature_softmax**（低算术强度归一化/reduce，前向计算主导区打不过，反向有优化结构可赢）。
- **带宽墙区（前向纯访存打不过）**：maxpool/l2norm_scale 前向（读写量=baseline）。l2norm 反向靠 chunk 融合真达标（§27）。
- **宿主分层区（融合密集，中弱宿主难赢）**：GEMM/online-softmax/causal_attn。
- **元规律**：归一化/reduce 家族**不是整体擦线或整体稳赢**——**前向由算术强度分野**（大 reduce/点积→真赢，单遍统计/逐元素→短核虚高），**反向普遍可赢**（Jacobian/缓存复用/列规约/chunk 融合）。

**主仓 layernorm case 异常修复（本次重估的触发源）**：回归时发现主仓 layernorm 反向 bench 出 **1228ms**（baseline）——根因是主仓 kernel 停在最早的**朴素灾难版**（3161fca，2.5 阶段建）：`layernorm_backward_dparam_kernel` 每个线程负责一列 d、沿 B 循环，**且每行内两个 `for k<D` 循环重算 mean/var**（同行 mean/var 被 D 个线程各重算一遍）→ **O(B·D²)**，LN_B=32768/D=1024 时 ~3.4 万亿次操作。§8/9 codex/手工优化到达标的版本（缓存 mean/inv_std + 二维分块列规约 + 前向 float4）**从没回收进主仓**（一直在 test_kt）。
- **修复**：把 test_kt 的诚实达标版（op.py + forward + backward）收进主仓——保留主仓 reference/config/__init__（接口一致、reference 数学等价）。反向 **O(B·D²)→O(B·D)**（1228ms→0.58ms），verify 前反向 5 种子全 PASS（误差 ~3e-6/dX、~1e-3/dgamma）。op.py 干净无计时特化（forward 存 mean/inv_std 供反向复用）。
- **但多规模复测揭示**：layernorm 前反向在计算主导区**都打不过**（LN_B=32768 前1.099/反1.065 擦线 PASS，放大 65536/131072 前掉 1.04/1.01、反掉 1.01/0.99 → SCALE_SUSPECT）——**§8/9"前1.08/反1.25达标"也是 LN_B=32768 短核虚高**（与 welford §15、temperature_softmax §24 同类，且 layernorm 整行小 reduce 算术强度最低，连反向都虚高）。
- **诚实结论**：异常（O(B·D²) 灾难）已修，case 恢复健康（正确 + O(B·D) 高效 + verify PASS）；但 layernorm 是"计算主导区前反向都打不过"的本征边界（低算术强度整行归一化），SCALE_SUSPECT 是诚实性能真相，非修复失败。主仓保留此达标版 kernel（短核规模数值正确、结构是列规约范例），性能结论按计算主导区加注。

**尝试把 layernorm 前反向优化到计算主导区真达标 → 坐实带宽墙本征边界（亲自穷尽）**：不满足于"短核虚高"的推断，实测尝试优化——
- **前向**：D=1024 路径每线程恰 1 个 float4（D4=256=BLOCK_SIZE），寄存器缓存 X 几乎免费（+1 float4/线程不压 occupancy，正是 loop.md"缓存量小才划算"的适用场景）。改成**统计与输出复用同一份寄存器缓存的 X（省第二遍全局读）**——理论减前向 ~1/3 访存。**实测 LN_B=131072：1.012→1.019×，几乎无提升**（verify 仍全 PASS）。省掉第二遍 X 读却无感 = 前向瓶颈不在"读两遍"，而是 candidate 访存量本就与 baseline 相同、都贴显存带宽上限。
- **带宽核算（判据）**：LN_B=131072/D=1024/fp32，前向最少访存 = 读X(512MB)+写Y(512MB)≈1GB，A100 ~1.5TB/s → 理论下界 ~0.67ms；candidate 实测 0.80ms = **达峰值带宽 84%**，与 baseline 0.81ms 持平。反向 candidate 2.08ms≈baseline 2.09ms、达理论 ~80%。**candidate 追平 torch.compile = 两者访存量相当 = 已达带宽墙**。
- **反向为何不硬试融合**：唯一理论空间是把 dparam 的第二遍 grad_y/X 重读融进 dX kernel（GroupNorm 反向 0.90→1.15 的手段）。但 layernorm dgamma/dbeta 是**逐列（D=1024）跨行规约**，与 dX 的行内规约维度正交，融合需 dX kernel（一 block 一行）对 D=1024 列各 atomicAdd → 高频 atomic 竞争爆炸（loop.md 翻车教训：per-element/高频 global atomicAdd 融合 0.85→0.26）。而 dX 已与 baseline 打平（主访存最优），融合省的第二遍会被 atomic 竞争吃掉——属"为擦 0.02× 硬堆手段翻车"，不试。
- **结论**：layernorm 前反向均为**带宽墙本征边界**（低算术强度整行归一化 D=1024，读写量=baseline、达峰值带宽 80%+，前向寄存器缓存实测无效为铁证）——与 **maxpool 前向同类**（§21）。回退无效的前向缓存改动（loop.md：无效即回退）。**这是诚实边界，非 skill/实现失败**：整行小 reduce 的归一化，前反向都无法比 torch.compile 融合访存更少。区别于 GroupNorm（大分组 reduce 算术强度够→前反向真赢）、cosine_sim/softmax_ce（点积/exp 规约→真赢）——**再次印证"归一化/reduce 前向由算术强度分野"：D=1024 整行是低强度端，撞带宽墙**。


## 30. 扩形态：GeGLU（第十九形态，融合逐元素算子链；反向融合真赢 + 前向带宽墙 + 三宿主全走确认闸门）

第十九形态。**纯自然语言意图**输入("把张量沿最后一维劈两半,一半当值、另一半过 GELU 当门,逐元素相乘",不含公式/shape),三宿主各走确认闸门。选它验证"融合逐元素算子链"——区别于 GroupNorm(reduce)、layernorm(reduce 带宽墙):GeGLU 是逐元素,但**反向要算 gelu(G)+gelu'(G) 两个非平凡函数**,融合有真价值。tanh 近似 GELU,X[B,T,2H] fp32,对 X 求梯度。

**① 确认闸门三宿主全触发 + 惊人一致**:codex/aider/gptme **都**推导出精确规格 + 列出相同的⚠️歧义澄清点(GELU exact/tanh 近似、劈半哪半是 value/gate、shape 展平否)+ PyTorch reference,停下等确认,未抢建 case。**三宿主都默认 exact erf GELU**——都被我纠正为 tanh 近似(GPT/LLaMA 实际用的 GeGLU)。codex 还**主动诚实预警**"纯逐元素带宽受限、torch.compile 也融合单 kernel、不承诺所有 shape 超过"——加固后的带宽墙认知内化生效。

**② 三宿主结果(我独立复验 + 多规模交叉)**:

| 宿主 | 前向 | 反向 | 结论 |
|------|------|------|------|
| **codex**(gpt-5.6-sol) | 1.02×(跨规模 B16/B32 稳,带宽墙) | **2.36~2.38×** 跨规模真达标 | ✅ 反向融合真赢,前向诚实报带宽墙(主动预警) |
| **aider**(GPT-5.5) | 1.00×(带宽墙,识别本征边界) | **3.93×**(f13e429) | ⚠️ 反向达标+识别边界,但硬试最后一招引入 `__tanhf`(应 tanhf)编译错,message 模式耗尽停在坏态(HEAD 58a267d 编译错,最佳版 f13e429) |
| **gptme**(GPT-5.5) | 1.01~1.02×(带宽墙,反复迭代未简洁认边界) | **1.23~1.25×** 跨轮稳达标 | ⚠️ 反向达标(但加速比低,因 reference 写法致 baseline 更强)+ 前向带宽墙;9+ 轮原地打转不果断停 |

**③ 核心结论**:

- **GeGLU 反向融合真赢(逐元素算子链的价值点)**:三宿主反向全达标(codex 2.36、aider 3.93、gptme 1.23)。反向要算 gelu(G) 和 gelu'(G) 两个非平凡函数,`torch.compile` 会物化它们的中间张量多趟访存;融合 kernel 一遍读 X、算 dV=dY·gelu(G)+dG=dY·V·gelu'(G) 写两半,省物化。**区别于 layernorm(reduce 反向也带宽墙)**——GeGLU 反向算术强度够(两个超越函数),是逐元素算子链**反向可靠融合真赢**的范例。

- **GeGLU 前向带宽墙(三宿主一致)**:纯逐元素(读 X 1.5GiB、写 Y),candidate 访存量=baseline、都达峰值带宽,三宿主前向一致 1.00~1.02×(codex 跨 B16/B32 稳、gptme 缩小规模到短核仍只 1.035)。**比归一化前向更硬的带宽墙**(连短核都擦不过 1.05)。codex 主动预警、aider 识别"本征边界"——加固的带宽墙认知在中强宿主生效。

- **reference 等价写法影响 baseline 强度(新发现)**:gptme 反向加速比(1.23)远低于 codex(2.36)/aider(3.93),但**三者 reference 都诚实**(tanh、向量化、CLEAN)。差异根源:gptme 用 `x.split(H,dim=-1)`、codex 用切片 `x[...,:h]`——**torch.compile 对 split 写法的反向融合更高效**,gptme 面对的 baseline 反向仅 2.4ms(codex/aider 是 7.7ms)。**同一算法的等价 reference 写法会让 torch.compile baseline 快慢不同,进而影响加速比**——都诚实,只是 baseline 强度不同。判达标看是否 >1.05(都过),不横向比宿主间加速比绝对值。

- **宿主收尾行为分层再现**:codex 干净停在达标态(前向诚实报边界);aider 识别边界却硬试最后一招引入编译错、message 模式耗尽停坏态(架构局限:非交互 message 模式无法无限迭代修错);gptme 反复迭代 9+ 轮不果断认边界(不如它 §28 GroupNorm 时的自主纠偏果断)。**skill 的带宽墙认知都内化了(都识别/预警),但收尾利落度依宿主架构/状态而异**。

**collected as 第十九形态参考 case**:codex 版收进主仓 `cases/geglu/`(tanh GeGLU + float4 反向 + autograd 封装 + bench.env;反向跨规模 2.36~2.38 真达标,前向带宽墙如实标注)。

**十九形态光谱更新**:稳赢区(RBF/scan/RoPE/linear_ssm/conv1d/gated_ssm/spmv/cosine_sim/softmax_ce/GroupNorm codex/**GeGLU 反向**)、前向带宽墙区(maxpool/l2norm/layernorm/welford/temp_softmax/**GeGLU 前向**——纯访存或低算术强度)、宿主分层区(GEMM/online-softmax/causal_attn)。**GeGLU 揭示逐元素算子链的前反向分野**:前向纯搬运=带宽墙,反向含多个超越函数=融合真赢。**元规律扩展**:算子能否赢看"candidate 能否比 torch.compile 访存更少"——reduce 大/点积/多超越函数反向→能(省物化);纯逐元素搬运前向→不能(访存量=baseline)。


## 31. 扩形态：grid_sample 双线性采样（第二十形态，数据依赖空间采样；三宿主前反向全真达标 + 确认闸门"点算子名"失效发现）

第二十形态。全新"**数据依赖空间采样**"维度:每个输出按 grid 归一化坐标 (x,y)∈[-1,1] 从 input 双线性插值 4 邻点,反向 grad_Y 按插值权重 atomicAdd scatter 回源 4 邻点。规格:input[N,C,H,W]+grid[N,OH,OW,2],**只对 input 求梯度**、zeros 边界、align_corners=False、bilinear。

**① 确认闸门"点算子名"失效发现(步骤 0.5 触发光谱的第二极)**:第一次给 codex 的意图里写了"**相当于 PyTorch 里的 grid_sample**"(点了明确算子名)→ **codex 直接跳过闸门开写,且擅自扩规格**(我定只对 input,codex 做成 input+grid 都求 + align_corners False/True 双分支)。**根因**:明确算子名让 codex 认为"PyTorch 默认语义已确定规格"无需确认。**这与 §24(9.4)是两极**:纯意图无信号→当聊天不进闸门;点明确算子名→认为规格已定跳过闸门;中间带(带任务信号+模糊意图)→正确进闸门。**验证**:给 aider/gptme 的意图**改用纯描述性语言、不点"grid_sample"**→ **两者都守住闸门**(推导规格+列歧义+停下问)——坐实"点算子名"是 codex 跳过的诱因。**已记 [[feedback-pure-algorithm-input]]**。处置:codex 重做严格只对 input。

**② 三宿主结果(独立复验,平衡计算主导区规模)**:

| 宿主 | 前向 | 反向 | 守规格 | 守闸门 |
|------|------|------|--------|--------|
| **codex**(gpt-5.6) | **2.11×** | **2.26×** | 重做后守(首次跳闸门扩 input+grid) | ✗ 首次(点算子名) |
| **aider**(GPT-5.5) | **1.14×** | **1.84×** | ✓ 只对 X | ✓ 纯描述守住 |
| **gptme**(GPT-5.5) | **1.09×** | **1.83×** | ✓ 最终只对 X(中途曾 input+grid,改名 gridsample 时修正) | ✓ 纯描述守住 |

**③ 核心结论**:

- **grid_sample 前反向都真赢(稳赢形态,全新采样维度)**:三宿主前反向全达标(平衡计算主导区规模,非短核)。融合优势真实——前向 candidate 一遍读 grid 坐标+4邻点插值写出,省 torch.compile 的中间 gather 物化;反向 atomic scatter 回源比 torch.compile 反向路径省。数据依赖空间采样算术强度够(4点插值+权重+坐标映射)。**是"数据依赖 gather + 反向 atomic scatter"维度的稳赢代表**(区别 scatter_add 的高冲突擦线——grid_sample 每点只散 4 邻点,冲突低)。

- **宿主 kernel 实现力分层(前向加速比)**:codex 2.11 > aider 1.14 > gptme 1.09。codex 前向 kernel 访存最优;aider/gptme 前向够达标但没抠到 codex 水平(gptme 迭代 9 轮才把前向从带宽墙附近抠到 1.09)。反向三宿主接近(2.26/1.84/1.83)。

- **规模挑选检测第三次实战生效(gptme)**:gptme 中途用短核规模(baseline 前 0.59ms)前向擦线 1.08 → harness 自动 ×2/×4 复测 → 判 **PASS_SCALE_SUSPECT** → gptme 响应后放大到计算主导区(GS_N=24/H=W=192,baseline 前 1.32ms)重测,前向真过 1.09。**检测引导 agent 从短核擦线走向计算主导区诚实达标**——正是加固目的。

- **规格遵循分层**:aider 全程守"只对 X";codex 首次擅自扩(input+grid,点算子名致)、重做守;gptme 中途 input+grid、改名时修正到只对 X。**明确说"只对 X"仍可能被 codex/gptme 首版扩宽**——对齐三宿主时须复验 grad_inputs 一致。

**④ 本轮暴露 harness 缺陷(auto-scale 病态形状)**:aider auto-test 卡 GPU7 环境,我复跑时 auto-scale 抓 config 首变量 GRIDSAMPLE_N 放大到 8192(H/W/OH/OW 不变)→ **病态形状**(batch 巨大、空间小)→ 前向 candidate 52ms vs baseline 13ms **假暴 0.256×**;换平衡大规模(N=64/C=64/H=W=128)→ 前 1.14× 真达标。**根因同 §28 GroupNorm 多维 OOM**:auto-scale 只放大多维 config 的第一个变量,对 N×C×H×W×OH×OW 这类多维 case 放大出病态形状/爆显存。**对策**:多维 case 靠 agent 建平衡 bench.env(codex/gptme 都建了、避开;aider 没建→撞上)。auto-scale 兜底应识别多维 config 等比放大而非单一变量放大(待加固)。

**collected as 第二十形态参考 case**:codex 版收进主仓 `cases/gridsample/`(bilinear+zeros+align_corners=False,只对 input,atomic scatter 反向;跨规模前 2.11/反 2.26 真达标)。

**二十形态光谱更新**:稳赢区(…/GeGLU 反向/**grid_sample 前反向**)。**grid_sample 归稳赢区**:数据依赖空间采样,前向融合省 gather 物化、反向低冲突 atomic scatter,前反向都赢。**元规律再证**:candidate 能比 torch.compile 访存更少即赢——grid_sample 前向省中间 gather 物化(赢)、反向 atomic scatter 直接回源(赢);区别 scatter_add 高冲突(atomic 竞争爆炸致擦线)——grid_sample 每点仅 4 邻点低冲突。确认闸门触发是光谱:太模糊(当聊天)/太明确点算子名(跳过)/中间带(正确进闸门)。
