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


