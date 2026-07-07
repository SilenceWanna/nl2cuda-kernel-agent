"""正确性验证器（评测核心，对被测实现只读）。

以 reference/rbf_reference.py 为金标准，验证任意"候选实现"的前向与反向。

候选实现的约定（统一接口）：
    impl(X, Y, gamma) -> K
      - X:[N,D], Y:[M,D] 是 requires_grad=True 的 leaf 张量
      - 返回 K:[N,M]
      - K.backward(G) 后，X.grad / Y.grad 必须被正确填充
    参考实现 rbf_kernel_reference 天然符合此约定；手写 CUDA kernel 用
    torch.autograd.Function 包装后也符合。

判据（任一不过即失败）：
    - 前向：≥5 组随机种子，allclose(atol=rtol=1e-2)
    - 反向：同种子、同 upstream 梯度，dX 与 dY 各自 allclose
    - 小规模：double 精度 torch.autograd.gradcheck 复核反向

用法：
    from tests.verify import verify_all
    from reference.rbf_reference import rbf_kernel_reference
    ok, report = verify_all(my_impl)          # 默认金标准=参考实现
    print(report)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from reference.config import CONFIG  # noqa: E402
from reference.rbf_reference import rbf_kernel_reference, make_inputs  # noqa: E402


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64,
            "float16": torch.float16}[name]


def verify_forward(impl, device="cuda", ref=rbf_kernel_reference):
    """前向 allclose：≥5 种子。返回 (ok, [每种子结果])。"""
    c = CONFIG
    dtype = _dtype(c.dtype)
    results = []
    for seed in c.seeds:
        Xi, Yi = make_inputs(c.N, c.M, c.D, dtype, device, seed, requires_grad=False)
        with torch.no_grad():
            K_ref = ref(Xi, Yi, c.gamma)
            K_imp = impl(Xi, Yi, c.gamma)
        ok = torch.allclose(K_imp, K_ref, atol=c.atol, rtol=c.rtol)
        max_abs = (K_imp - K_ref).abs().max().item()
        results.append((seed, ok, max_abs))
    return all(r[1] for r in results), results


def verify_backward(impl, device="cuda", ref=rbf_kernel_reference):
    """反向 allclose：同种子、同 upstream 梯度 G，比较 dX/dY。返回 (ok, [结果])。"""
    c = CONFIG
    dtype = _dtype(c.dtype)
    results = []
    for seed in c.seeds:
        # 参考路径
        Xr, Yr = make_inputs(c.N, c.M, c.D, dtype, device, seed, requires_grad=True)
        Kr = ref(Xr, Yr, c.gamma)
        g = torch.Generator(device=device).manual_seed(seed + 10_000)
        G = torch.randn(c.N, c.M, dtype=dtype, device=device, generator=g)
        Kr.backward(G)
        dXr, dYr = Xr.grad.detach().clone(), Yr.grad.detach().clone()

        # 候选路径（相同输入、相同 G）
        Xi, Yi = make_inputs(c.N, c.M, c.D, dtype, device, seed, requires_grad=True)
        Ki = impl(Xi, Yi, c.gamma)
        Ki.backward(G.clone())
        dXi, dYi = Xi.grad, Yi.grad

        okX = torch.allclose(dXi, dXr, atol=c.atol, rtol=c.rtol)
        okY = torch.allclose(dYi, dYr, atol=c.atol, rtol=c.rtol)
        mX = (dXi - dXr).abs().max().item()
        mY = (dYi - dYr).abs().max().item()
        results.append((seed, okX and okY, mX, mY))
    return all(r[1] for r in results), results


def verify_gradcheck(impl, device="cuda"):
    """小规模 double 精度 gradcheck 复核反向。返回 (ok, msg)。"""
    c = CONFIG
    Xg, Yg = make_inputs(c.grad_N, c.grad_M, c.grad_D, torch.float64, device,
                         seed=0, requires_grad=True)
    try:
        ok = torch.autograd.gradcheck(
            lambda X, Y: impl(X, Y, c.gamma), (Xg, Yg),
            atol=1e-4, rtol=1e-3, raise_exception=True,
        )
        return ok, "gradcheck passed"
    except Exception as e:  # noqa: BLE001
        return False, f"gradcheck failed: {e}"


def verify_all(impl, device=None, ref=rbf_kernel_reference):
    """完整验证，返回 (ok, report_str)。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    lines = [f"=== 正确性验证 (device={device}) ===", CONFIG.__class__.__name__ + ": "]
    from reference.config import summary
    lines.append(summary())

    fwd_ok, fwd = verify_forward(impl, device, ref)
    lines.append(f"\n[前向] {'PASS' if fwd_ok else 'FAIL'}")
    for seed, ok, m in fwd:
        lines.append(f"  seed={seed}: {'ok' if ok else 'X'}  max_abs_err={m:.3e}")

    bwd_ok, bwd = verify_backward(impl, device, ref)
    lines.append(f"\n[反向] {'PASS' if bwd_ok else 'FAIL'}")
    for seed, ok, mX, mY in bwd:
        lines.append(f"  seed={seed}: {'ok' if ok else 'X'}  "
                     f"dX_err={mX:.3e} dY_err={mY:.3e}")

    gc_ok, gc_msg = verify_gradcheck(impl, device)
    lines.append(f"\n[gradcheck] {'PASS' if gc_ok else 'FAIL'}: {gc_msg}")

    overall = fwd_ok and bwd_ok and gc_ok
    lines.append(f"\n=== 总判定: {'PASS' if overall else 'FAIL'} ===")
    return overall, "\n".join(lines)


if __name__ == "__main__":
    # 自检：用参考实现验证参考实现自身，应当全 PASS（误差 ~0）。
    ok, report = verify_all(rbf_kernel_reference)
    print(report)
    sys.exit(0 if ok else 1)
