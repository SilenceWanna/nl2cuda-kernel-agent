"""计时基准器（防测量偏差 / 防作弊）。

⚠️ 本脚本对 agent **只读**、进程隔离、不得修改。计时协议直接对应工作目标第五节。

计时协议：
- 同一 GPU、同一输入；单块本地 NVIDIA GPU。
- warmup ≥ 10；正式测量 ≥ 100 次取**几何均值**。
- 每次计时前后 torch.cuda.synchronize()，用 CUDA events 计时。
- 3 次重跑控制方差，CV > 5% 作废重测。
- **前向、反向分别计时**并各自达标。
- 达标：自定义 kernel 相对 torch.compile(默认 mode) 前反向均快 ≥5%（≥1.05x）。

被测对象的统一接口：
- 前向计时：传入 forward_fn(X, Y, gamma) -> K，只计前向。
- 反向计时：传入 backward_prep -> (loss_or_K, grads_tuple)；反复只计 K.backward(G)。
  为公平，反向计时不含前向：每轮先重建计算图再计时 backward（见 time_backward）。

候选 vs baseline：
- baseline = torch.compile(rbf_kernel_reference, mode 默认)。
- 候选 = 手写 kernel（阶段1）或 skill 生成 kernel（阶段2+），须用 autograd.Function 包装。
"""

import sys
import os
import math
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from reference.config import CONFIG  # noqa: E402
from reference.rbf_reference import rbf_kernel_reference, make_inputs  # noqa: E402


def _dtype(name):
    return {"float32": torch.float32, "float64": torch.float64,
            "float16": torch.float16}[name]


def _geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def _cuda_time_once(fn):
    """用 CUDA events 计一次 fn() 的耗时（ms）。fn 内部不应含 host 同步。"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)  # ms


def _measure(fn, warmup, iters):
    """warmup 后测 iters 次，返回耗时列表（ms）。整个 fn() 计入计时区。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        times.append(_cuda_time_once(fn))
    return times


def _measure_backward(build_graph, warmup, iters):
    """反向专用计时：每轮在计时区外重建计算图，只对 backward 计时。

    build_graph(): 运行前向（带 grad），返回一个零参 backward 闭包 do_bwd()。
    这样前向（建图）不计入计时，且每轮用全新的图——对 torch.compile 与
    自定义 autograd.Function 都公平，无需 retain_graph。
    """
    for _ in range(warmup):
        do_bwd = build_graph()
        do_bwd()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        do_bwd = build_graph()          # 计时区外：前向建图
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        do_bwd()                        # 计时区内：只有 backward
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _forward_geomean_cv(forward_fn, X, Y, gamma, warmup=None, iters=None,
                        repeats=None, cv_threshold=None):
    """前向计时：重跑 repeats 次几何均值，算 CV。整个前向计入计时区（no_grad）。"""
    c = CONFIG
    warmup = c.warmup if warmup is None else warmup
    iters = c.iters if iters is None else iters
    repeats = c.repeats if repeats is None else repeats
    cv_threshold = c.cv_threshold if cv_threshold is None else cv_threshold

    Xl, Yl = X.detach(), Y.detach()

    def step():
        with torch.no_grad():
            forward_fn(Xl, Yl, gamma)

    geomeans = [_geomean(_measure(step, warmup, iters)) for _ in range(repeats)]
    return _summarize("forward", geomeans, cv_threshold)


def _backward_geomean_cv(forward_fn, X, Y, gamma, G, warmup=None, iters=None,
                         repeats=None, cv_threshold=None):
    """反向计时：每轮计时区外重建图，只对 backward 计时。前向不计入。

    对 torch.compile 与自定义 autograd.Function 都公平（每轮全新图，无 retain_graph）。
    """
    c = CONFIG
    warmup = c.warmup if warmup is None else warmup
    iters = c.iters if iters is None else iters
    repeats = c.repeats if repeats is None else repeats
    cv_threshold = c.cv_threshold if cv_threshold is None else cv_threshold

    def build_graph():
        Xl = X.detach().clone().requires_grad_(True)
        Yl = Y.detach().clone().requires_grad_(True)
        K = forward_fn(Xl, Yl, gamma)

        def do_bwd():
            K.backward(G)
        return do_bwd

    geomeans = [_geomean(_measure_backward(build_graph, warmup, iters))
                for _ in range(repeats)]
    return _summarize("backward", geomeans, cv_threshold)


def _summarize(kind, geomeans, cv_threshold):
    mean = statistics.mean(geomeans)
    cv = (statistics.pstdev(geomeans) / mean) if mean > 0 else float("inf")
    return {
        "kind": kind,
        "geomeans_ms": geomeans,
        "geomean_ms": _geomean(geomeans),
        "cv": cv,
        "ok": cv <= cv_threshold,
    }


def compare(candidate_fn, device=None, seed=0):
    """对比候选实现与 torch.compile baseline 的前反向。

    candidate_fn(X,Y,gamma)->K : 候选实现。前向计时在 no_grad 下调用它；
      反向计时用它建图后只计 backward。手写 kernel 须用 autograd.Function 包装，
      使同一函数既能前向、又能被 backward。
    返回 (report_str, pass_bool)。
    """
    c = CONFIG
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "计时必须在 CUDA 上进行"
    dtype = _dtype(c.dtype)

    X, Y = make_inputs(c.N, c.M, c.D, dtype, device, seed, requires_grad=False)
    g = torch.Generator(device=device).manual_seed(777)
    G = torch.randn(c.N, c.M, dtype=dtype, device=device, generator=g)

    baseline_fn = torch.compile(rbf_kernel_reference)  # 默认 mode

    lines = ["=== 计时基准 (CUDA events, 几何均值) ===", CONFIG_summary()]

    results = {}
    for name, fn in [("baseline(torch.compile)", baseline_fn),
                     ("candidate", candidate_fn)]:
        fres = _forward_geomean_cv(fn, X, Y, c.gamma)
        bres = _backward_geomean_cv(fn, X, Y, c.gamma, G)
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
    lines.append(f"  forward : {fwd_speedup:.4f}x  "
                 f"(需 ≥{c.speedup_target}x) {'PASS' if fwd_speedup >= c.speedup_target else 'FAIL'}")
    lines.append(f"  backward: {bwd_speedup:.4f}x  "
                 f"(需 ≥{c.speedup_target}x) {'PASS' if bwd_speedup >= c.speedup_target else 'FAIL'}")

    cv_ok = all(r["ok"] for r in (bf, bb, cf, cb))
    passed = (fwd_speedup >= c.speedup_target and bwd_speedup >= c.speedup_target and cv_ok)
    if not cv_ok:
        lines.append("  ⚠️ 存在 CV>5%，结果作废需重测")
    lines.append(f"\n=== 达标判定: {'PASS' if passed else 'FAIL'} ===")
    return "\n".join(lines), passed


def CONFIG_summary():
    from reference.config import summary
    return summary()


if __name__ == "__main__":
    # 自检：候选=参考实现自身（未编译）对比 baseline=torch.compile(参考)。
    # 预期 baseline 更快（编译优化），故 candidate 加速比 <1，判 FAIL——
    # 这只是验证计时管线可运行，不是真实达标测试。
    if not torch.cuda.is_available():
        print("需要 CUDA GPU 才能计时。本地无 GPU，请在 Colab 运行。")
        sys.exit(0)
    report, passed = compare(rbf_kernel_reference)
    print(report)
