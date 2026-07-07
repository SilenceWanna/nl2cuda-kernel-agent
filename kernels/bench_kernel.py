"""1.6 首次基准：手写 RBF kernel（朴素版）vs torch.compile。

用 benchmarks/bench.py 的 compare()，候选 = rbf_autograd（前反向 kernel）。
朴素 kernel 此时大概率还打不过 torch.compile（优化留到阶段3），本步用于：
- 确认计时管线能跑真实 kernel；
- 得到朴素版 vs baseline 的差距，指导阶段3优化方向。

用法（Colab）：python kernels/bench_kernel.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from benchmarks.bench import compare  # noqa: E402
from kernels.rbf_op import rbf_autograd  # noqa: E402


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("需要 CUDA GPU。请在 Colab 运行。")
        sys.exit(0)

    report, passed = compare(rbf_autograd)
    print(report)
    # 首次基准不要求达标，退出码固定 0（仅报告差距）。
    sys.exit(0)
