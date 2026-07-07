"""通用正确性验证器（算法无关）。以 case 的参考实现为金标准，验证候选实现。

候选实现统一接口：impl(inputs: dict, params: dict) -> output
  - inputs 中 case.grad_inputs 列出的张量为 requires_grad=True 的 leaf；
  - output.backward(G) 后，这些张量的 .grad 被填充。

判据（任一不过即失败）：
  - 前向：≥5 组随机种子，allclose(atol=rtol=1e-2)
  - 反向：同种子、同 upstream 梯度 G，对每个 grad_input 的梯度各自 allclose

⚠️ 本文件属评测基座，对 agent 只读，不得修改。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from framework.protocol import PROTOCOL, dtype_of  # noqa: E402


def _clone_requires_grad(inputs, grad_inputs):
    """复制输入 dict；对 grad_inputs 中的张量置 requires_grad。"""
    out = {}
    for k, v in inputs.items():
        t = v.detach().clone()
        if k in grad_inputs:
            t.requires_grad_(True)
        out[k] = t
    return out


def verify_forward(case, impl, device="cuda"):
    """前向 allclose：≥5 种子。返回 (ok, [每种子结果])。"""
    p = PROTOCOL
    dtype = dtype_of(case.dtype)
    results = []
    for seed in p.seeds:
        inputs = case.make_inputs(seed, dtype, device, requires_grad=False)
        with torch.no_grad():
            out_ref = case.reference_forward(inputs, case.params)
            out_imp = impl(inputs, case.params)
        ok = torch.allclose(out_imp, out_ref, atol=p.atol, rtol=p.rtol)
        max_abs = (out_imp - out_ref).abs().max().item()
        results.append((seed, ok, max_abs))
    return all(r[1] for r in results), results


def verify_backward(case, impl, device="cuda"):
    """反向 allclose：同种子、同 upstream G，对每个 grad_input 比较梯度。"""
    p = PROTOCOL
    dtype = dtype_of(case.dtype)
    results = []
    for seed in p.seeds:
        base = case.make_inputs(seed, dtype, device, requires_grad=False)

        # 参考路径
        ref_in = _clone_requires_grad(base, case.grad_inputs)
        out_ref = case.reference_forward(ref_in, case.params)
        g = torch.Generator(device=device).manual_seed(seed + 10_000)
        G = torch.randn(out_ref.shape, dtype=dtype, device=device, generator=g)
        out_ref.backward(G)
        ref_grads = {k: ref_in[k].grad.detach().clone() for k in case.grad_inputs}

        # 候选路径（相同输入、相同 G）
        imp_in = _clone_requires_grad(base, case.grad_inputs)
        out_imp = impl(imp_in, case.params)
        out_imp.backward(G.clone())

        per_input_ok = {}
        per_input_err = {}
        for k in case.grad_inputs:
            gi = imp_in[k].grad
            gr = ref_grads[k]
            per_input_ok[k] = torch.allclose(gi, gr, atol=p.atol, rtol=p.rtol)
            per_input_err[k] = (gi - gr).abs().max().item()
        ok = all(per_input_ok.values())
        results.append((seed, ok, per_input_err))
    return all(r[1] for r in results), results


def verify_all(case, impl, device=None):
    """完整验证，返回 (ok, report_str)。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    lines = [f"=== 正确性验证: case={case.name} (device={device}) ===",
             f"grad_inputs={case.grad_inputs} params={case.params} dtype={case.dtype}"]

    fwd_ok, fwd = verify_forward(case, impl, device)
    lines.append(f"\n[前向] {'PASS' if fwd_ok else 'FAIL'}")
    for seed, ok, m in fwd:
        lines.append(f"  seed={seed}: {'ok' if ok else 'X'}  max_abs_err={m:.3e}")

    bwd_ok, bwd = verify_backward(case, impl, device)
    lines.append(f"\n[反向] {'PASS' if bwd_ok else 'FAIL'}")
    for seed, ok, errs in bwd:
        errstr = " ".join(f"d{k}_err={v:.3e}" for k, v in errs.items())
        lines.append(f"  seed={seed}: {'ok' if ok else 'X'}  {errstr}")

    overall = fwd_ok and bwd_ok
    lines.append(f"\n=== 总判定: {'PASS' if overall else 'FAIL'} ===")
    return overall, "\n".join(lines)
