"""通用计时基准器（算法无关，防测量偏差 / 防作弊）。

候选实现统一接口：candidate(inputs: dict, params: dict) -> output
baseline = torch.compile(case.reference_forward)（默认 mode）。

计时协议见 framework/protocol.py：warmup≥10、正式≥100 取几何均值、CUDA events、
每次前后 synchronize、3 次重跑算 CV、CV>5% 作废重测、前反向分别计时、达标 1.05x。

反向计时：每轮在计时区外重建计算图，只对 backward 计时（对 torch.compile 与
自定义 autograd.Function 都公平，无 retain_graph）。

⚠️ 本文件属评测基座，对 agent 只读，不得修改。
"""

import sys
import os
import math
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from framework.protocol import PROTOCOL, dtype_of  # noqa: E402


def _geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def _cuda_time_once(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def _measure(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    return [_cuda_time_once(fn) for _ in range(iters)]


def _measure_backward(build_graph, warmup, iters):
    """每轮计时区外重建图，只对 backward 计时。"""
    for _ in range(warmup):
        build_graph()()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        do_bwd = build_graph()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        do_bwd()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _summarize(kind, geomeans, cv_threshold):
    mean = statistics.mean(geomeans)
    cv = (statistics.pstdev(geomeans) / mean) if mean > 0 else float("inf")
    return {"kind": kind, "geomeans_ms": geomeans, "geomean_ms": _geomean(geomeans),
            "cv": cv, "ok": cv <= cv_threshold}


def _forward_geomean_cv(fn, inputs, params):
    p = PROTOCOL
    detached = {k: v.detach() for k, v in inputs.items()}

    def step():
        with torch.no_grad():
            fn(detached, params)

    geomeans = [_geomean(_measure(step, p.warmup, p.iters)) for _ in range(p.repeats)]
    return _summarize("forward", geomeans, p.cv_threshold)


def _backward_geomean_cv(fn, inputs, params, grad_inputs, G):
    p = PROTOCOL

    def build_graph():
        loc = {}
        for k, v in inputs.items():
            t = v.detach().clone()
            if k in grad_inputs:
                t.requires_grad_(True)
            loc[k] = t
        out = fn(loc, params)

        def do_bwd():
            out.backward(G)
        return do_bwd

    geomeans = [_geomean(_measure_backward(build_graph, p.warmup, p.iters))
                for _ in range(p.repeats)]
    return _summarize("backward", geomeans, p.cv_threshold)


def compare(case, candidate, device=None, seed=0):
    """对比候选实现与 torch.compile baseline 的前反向。返回 (report_str, passed)。"""
    p = PROTOCOL
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "计时必须在 CUDA 上进行"
    dtype = dtype_of(case.dtype)

    inputs = case.make_inputs(seed, dtype, device, requires_grad=False)
    # upstream 梯度 G：按参考输出形状生成
    with torch.no_grad():
        out_shape = case.reference_forward(inputs, case.params).shape
    g = torch.Generator(device=device).manual_seed(777)
    G = torch.randn(out_shape, dtype=dtype, device=device, generator=g)

    baseline = torch.compile(case.reference_forward)

    lines = [f"=== 计时基准: case={case.name} (CUDA events, 几何均值) ===",
             f"grad_inputs={case.grad_inputs} params={case.params} dtype={case.dtype}"]

    results = {}
    for name, fn in [("baseline(torch.compile)", baseline), ("candidate", candidate)]:
        fres = _forward_geomean_cv(fn, inputs, case.params)
        bres = _backward_geomean_cv(fn, inputs, case.params, case.grad_inputs, G)
        results[name] = (fres, bres)
        lines.append(f"\n[{name}]")
        lines.append(f"  forward : {fres['geomean_ms']:.4f} ms  "
                     f"CV={fres['cv']*100:.2f}%  {'ok' if fres['ok'] else 'CV超阈值!'}")
        lines.append(f"  backward: {bres['geomean_ms']:.4f} ms  "
                     f"CV={bres['cv']*100:.2f}%  {'ok' if bres['ok'] else 'CV超阈值!'}")

    (bf, bb) = results["baseline(torch.compile)"]
    (cf, cb) = results["candidate"]
    fwd_speedup = bf["geomean_ms"] / cf["geomean_ms"]
    bwd_speedup = bb["geomean_ms"] / cb["geomean_ms"]

    lines.append(f"\n[加速比 candidate vs baseline]")
    lines.append(f"  forward : {fwd_speedup:.4f}x  (需 ≥{p.speedup_target}x) "
                 f"{'PASS' if fwd_speedup >= p.speedup_target else 'FAIL'}")
    lines.append(f"  backward: {bwd_speedup:.4f}x  (需 ≥{p.speedup_target}x) "
                 f"{'PASS' if bwd_speedup >= p.speedup_target else 'FAIL'}")

    cv_ok = all(r["ok"] for r in (bf, bb, cf, cb))
    passed = (fwd_speedup >= p.speedup_target and bwd_speedup >= p.speedup_target and cv_ok)
    if not cv_ok:
        lines.append("  ⚠️ 存在 CV>5%，结果作废需重测")
    lines.append(f"\n=== 达标判定: {'PASS' if passed else 'FAIL'} ===")
    return "\n".join(lines), passed
