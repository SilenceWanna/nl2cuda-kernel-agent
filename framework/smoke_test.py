"""编译链路冒烟测试：在 Colab(T4) 上编译 smoke.cu 并验证 vadd 正确。

用法：
    python framework/smoke_test.py

通过 = nvcc + torch cpp_extension + sm_75 编译链路 OK，可以放心写真正的 kernel。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from framework.loader import load_kernel  # noqa: E402


def main():
    if not torch.cuda.is_available():
        print("需要 CUDA GPU。本地无 GPU，请在 Colab 运行。")
        return 0

    print("编译 smoke.cu (sm_75) ...")
    mod = load_kernel("smoke_vadd", ["smoke.cu"])
    print("编译成功。")

    n = 1_000_003
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    out = mod.vadd(a, b)
    torch.cuda.synchronize()

    ref = a + b
    ok = torch.allclose(out, ref, atol=1e-6, rtol=1e-6)
    max_err = (out - ref).abs().max().item()
    print(f"vadd 正确性: {'PASS' if ok else 'FAIL'}  max_abs_err={max_err:.3e}  n={n}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
