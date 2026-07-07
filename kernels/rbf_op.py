"""RBF kernel 的 Python 封装（阶段1）。

阶段1按 1.2→1.5 逐步搭建：
- 现在（1.2/1.3）：只有前向 kernel。提供 forward_only 供前向 allclose 对拍。
- 之后（1.4/1.5）：补反向 kernel，用 torch.autograd.Function 包装成完整实现，
  可直接喂给 tests/verify.py 的 verify_all（前向+反向+gradcheck）。
"""

import sys
import os
import functools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from kernels.loader import load_kernel  # noqa: E402


@functools.lru_cache(maxsize=1)
def _fwd_module():
    """编译并缓存前向 kernel 扩展（进程内只编译一次）。"""
    return load_kernel("rbf_forward", ["rbf_forward.cu"])


def forward_only(X, Y, gamma):
    """直接调用前向 kernel，返回 K:[N,M]。无 autograd（仅供前向正确性对拍）。"""
    return _fwd_module().rbf_forward(X, Y, float(gamma))


if __name__ == "__main__":
    # 前向正确性自检：用 tests/verify.py 的 verify_forward 对拍参考实现。
    if not torch.cuda.is_available():
        print("需要 CUDA GPU。请在 Colab 运行。")
        sys.exit(0)

    from tests.verify import verify_forward
    from reference.rbf_reference import rbf_kernel_reference

    print("编译并运行前向 kernel，对拍参考实现 ...")
    ok, results = verify_forward(forward_only, device="cuda", ref=rbf_kernel_reference)
    print(f"\n[前向 allclose] {'PASS' if ok else 'FAIL'}")
    for seed, seed_ok, max_abs in results:
        print(f"  seed={seed}: {'ok' if seed_ok else 'X'}  max_abs_err={max_abs:.3e}")
    sys.exit(0 if ok else 1)
