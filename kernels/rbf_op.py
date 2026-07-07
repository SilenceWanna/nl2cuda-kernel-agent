"""RBF kernel 的 Python 封装（阶段1）。

- 前向 kernel: rbf_forward.cu  → forward_only（仅前向对拍用）
- 反向 kernel: rbf_backward.cu → 与前向一起用 torch.autograd.Function 包装为
  rbf_autograd，符合 tests/verify.py 的统一实现接口：
      rbf_autograd(X, Y, gamma) -> K, 且 K.backward(G) 后 X.grad/Y.grad 被填充。

精度说明（重要）：本 kernel 是 **float32-only**（验收精度即 fp32，禁止降精度换速度）。
tests/verify.py 的小规模 gradcheck 用 double 扰动，float32 kernel 无法直接参与，
故 kernel 的反向正确性以**对拍参考 autograd 的 allclose（≥5 种子，验收形状）**为准
——这与验收判据（allclose atol=rtol=1e-2）完全一致，且参考实现自身已通过 gradcheck，
等价性有保证。
"""

import sys
import os
import functools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from kernels.loader import load_kernel  # noqa: E402


@functools.lru_cache(maxsize=1)
def _fwd_module():
    return load_kernel("rbf_forward", ["rbf_forward.cu"])


@functools.lru_cache(maxsize=1)
def _bwd_module():
    return load_kernel("rbf_backward", ["rbf_backward.cu"])


def forward_only(X, Y, gamma):
    """直接调用前向 kernel，返回 K:[N,M]。无 autograd（仅供前向正确性对拍）。"""
    return _fwd_module().rbf_forward(X, Y, float(gamma))


class RBFFunction(torch.autograd.Function):
    """前向 kernel + 反向 kernel 组成的自定义 autograd 算子（float32）。"""

    @staticmethod
    def forward(ctx, X, Y, gamma):
        K = _fwd_module().rbf_forward(X, Y, float(gamma))
        ctx.save_for_backward(X, Y)
        ctx.gamma = float(gamma)
        return K

    @staticmethod
    def backward(ctx, G):
        X, Y = ctx.saved_tensors
        dX, dY = _bwd_module().rbf_backward(X, Y, G.contiguous(), ctx.gamma)
        # 对应 forward 的三个输入 (X, Y, gamma)：gamma 不需要梯度
        return dX, dY, None


def rbf_autograd(X, Y, gamma):
    """完整实现（前向+反向），符合 verify.py 接口。X/Y 应 requires_grad。"""
    return RBFFunction.apply(X, Y, gamma)


if __name__ == "__main__":
    # 完整正确性自检：前向 + 反向 allclose 对拍参考实现（≥5 种子）。
    if not torch.cuda.is_available():
        print("需要 CUDA GPU。请在 Colab 运行。")
        sys.exit(0)

    from tests.verify import verify_forward, verify_backward
    from reference.rbf_reference import rbf_kernel_reference

    print("编译前向+反向 kernel，对拍参考实现 ...")

    fok, fres = verify_forward(rbf_autograd, device="cuda", ref=rbf_kernel_reference)
    print(f"\n[前向 allclose] {'PASS' if fok else 'FAIL'}")
    for seed, ok, m in fres:
        print(f"  seed={seed}: {'ok' if ok else 'X'}  max_abs_err={m:.3e}")

    bok, bres = verify_backward(rbf_autograd, device="cuda", ref=rbf_kernel_reference)
    print(f"\n[反向 allclose] {'PASS' if bok else 'FAIL'}")
    for seed, ok, mX, mY in bres:
        print(f"  seed={seed}: {'ok' if ok else 'X'}  dX_err={mX:.3e} dY_err={mY:.3e}")

    overall = fok and bok
    print(f"\n=== 前反向正确性: {'PASS' if overall else 'FAIL'} ===")
    sys.exit(0 if overall else 1)
