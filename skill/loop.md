# 优化迭代循环（loop）

当 kernel 已正确但**未达速度标**（前向或反向 <1.05×）时，按本循环迭代优化。宿主 agent 或自动 loop 驱动都可执行。**算法无关**：对任意 case 通用。

## 循环体（每轮）

```
1. bench   → python skill/scripts/bench_case.py --case <name>
             读当前前/反向加速比、baseline 绝对耗时、CV。
2. 诊断    → 定位瓶颈（见下"优化手段"），选一项改动。
3. 改 kernel → 只改 cases/<name>/kernels/*.cu（及 op.py 封装），不动 framework/。
4. verify  → python skill/scripts/verify_case.py --case <name>
             必须仍前反向全 PASS（allclose）。不过则回退本轮改动。
5. bench   → 重新计时，记录加速比是否提升。
```

## 终止条件（满足任一即停）

- ✅ **达标**：前向、反向加速比均 ≥1.05×，且 3 次重跑 CV≤5%，正确性仍全 PASS。→ 进入交付。
- ⏹ **轮次上限**：达到预设最大轮次（建议 8–12 轮）仍未达标 → 停下，报告当前最好结果与瓶颈分析，请人工介入（可能需换优化策略或升级 GPU）。
- ⏹ **收益枯竭**：连续 2–3 轮无有效提升（且已排除测量噪声）→ 停下报告。

## 优化手段（按预期收益排序，据 profile 选择）

1. **shared-memory tiling**：一个 block 协作把输入分块载入 shared memory，块内复用，减少全局内存重复读。通常是最大收益项。
2. **前向缓存复用**：反向若重算了前向的中间量（如 RBF 反向重算 dist/K），改为前向保存、反向读取，省重复计算。
3. **float4 向量化**：D 维连续访存用 `float4` 一次读 4 个，提升带宽利用与访存合并。
4. **warp 原语**：规约类用 `__shfl_down_sync` 等 warp 内规约，替代 shared memory 往返。
5. **算子融合**：把多步（如距离→exp）融进一个 kernel，避免中间张量物化与额外 kernel launch。
6. **线程/块配置调优**：block 大小、每线程工作量（thread coarsening）、occupancy 平衡。

## 纪律

- **正确性优先**：任何优化后必须先过 verify 再看 bench；不过则丢弃该轮改动。
- **CV 门禁**：bench 报 CV>5% 时结果作废，重测；不要基于噪声数据做优化决策。
- **前反向都要达标**：只赢前向或只赢反向都不算达成，继续攻未达标的一侧。
- **不碰评测基座**：`framework/` 只读；优化只在 case 的 kernel/op 内进行。
- **不降精度**：除非算法描述指定低精度，否则保持 fp32；不得用 fast-math/降精度换速度。
