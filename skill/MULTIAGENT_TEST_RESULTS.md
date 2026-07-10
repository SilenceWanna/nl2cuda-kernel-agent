# 多-Agent 宿主无关性测试结果

本 skill 的核心主张是**宿主无关**：方法论（`SKILL.md`）+ 独立 python 工具（`framework/`、`skill/scripts/`），
不绑定任何特定 agent。本文档记录在多个现成 agent 上驱动 skill 完成同一任务的实测结果。

## 测试设置

- **任务**：给定 `cases/softmax_ce/description.md`（Softmax 交叉熵损失的自然语言描述），让 agent 仅凭
  `SKILL.md` 方法论，从零写出 PyTorch 参考 + CUDA 前反向 kernel + Case 封装，通过正确性验证并超过 torch.compile。
- **做法**：每个 agent 用独立测试副本，删掉 softmax_ce 实现只留 `description.md`，喂统一提示让其自主实现。
- **模型**：均经京东内网 OpenAI 兼容代理（多数用 `GPT-5.5-joybuilder`，goose 用 `DeepSeek-V3`）。
- **验收**：产物推到独立分支 → 在 A100(sm_80) 上跑 `verify_case.py`（前反向 allclose）+ `bench_case.py`（vs torch.compile）。

## 结果：4 个宿主全部达标

| 宿主 | 形态 | 正确性 | 性能(vs torch.compile) | 备注 |
|------|------|--------|------------------------|------|
| **Claude Code** | CLI | ✅ 前反向 PASS | 达标（对照基准） | 项目主宿主 |
| **codex** | CLI/桌面 | ✅ 前反向 PASS | 前 1.97× / 反 1.80× | 接口一次对（提示读了 framework/case.py） |
| **aider** | CLI | ✅ 前反向 PASS | 前 1.64× / 反 1.60× | 先猜错接口 → **改进 SKILL.md 后自愈** |
| **gptme** | CLI (WSL) | ✅ 前反向 PASS | 前 1.64× / 反 1.46× | 接口一次对；一次编译错自愈（CUDART_INF_F→FLT_MAX） |

softmax_ce 三份 agent 实现分别在 `test/codex-softmax`、`test/aider-softmax`、`test/gptme-softmax` 分支。

## 关键发现：文档质量决定 agent 一次成功率

- **codex / gptme**：被提示读 `framework/case.py`（或读到含"Case 协议"章节的 SKILL.md）→ 接口一次全对。
- **aider**：初次用旧版 SKILL.md（无显式接口清单）→ 靠猜写 Case，缺必填字段、make_inputs 返回二元组 → verify 崩溃。
  据此**两轮改进 SKILL.md**（新增"Case 协议"章节：7 必填字段表 + `__init__.py`/`reference.py` 骨架 +
  "make_inputs 只返回 dict"约定 + "别猜直接读 case.py"）。aider 重读后**一次改对**，最终达标。

→ **结论**：skill 的可迁移性靠的是把接口契约讲清楚。补强文档后，不同 agent 都能顺利完成。

## 环境受限、未跑成的宿主（均与 skill 无关）

| 宿主 | 受阻原因 |
|------|---------|
| **goose** | GPT-5.5 撞 temperature 限制；换 DeepSeek-V3 后其工具调用格式与 goose 不兼容（模型输出原生 tool token 未转成 OpenAI tool_calls） |
| **Cline** | 同时发 temperature + content 数组；京东代理无"两者都接受"的已授权模型（DeepSeek 接受 temperature 但拒 content 数组，GPT-5.5 反之）——死结 |
| **OpenHands** | 强依赖 Docker：本机 Docker 引擎频繁抖动 + 公司网络拉不动 ghcr 大镜像（app/runtime，换多个国内代理源均被网络挡）→ 部署不可行 |

这三者卡的都是**宿主自身与京东网关 / 本机 Docker / 公司网络的适配问题**，不是 skill 能力问题。
反而佐证：**只要宿主能正常连模型 + 跑 python，skill 就能用。**

## 总结

skill 宿主无关性得到充分验证（4 个成功宿主，涵盖 CLI/桌面/WSL 多种形态，GPT-5.5/DeepSeek 多个模型）。
失败的 3 个宿主均因环境/网关兼容问题受阻，与 skill 设计无关。多-agent 测试同时驱动了 SKILL.md 的实质改进
（Case 协议章节），使 skill 对未来任意宿主更健壮。
