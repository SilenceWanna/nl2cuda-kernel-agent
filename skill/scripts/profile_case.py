"""通用 profiling 工具：分解一个 case 候选实现前/反向的耗时，
区分 CUDA kernel 本体时间 vs CPU 端开销（分配/launch/dispatch）。

用法：
    python skill/scripts/profile_case.py --case rbf [--impl module:fn]

输出 torch.profiler 的 key_averages 表（按 CUDA 时间排序），帮助定位真瓶颈：
- 若某个 kernel 占绝大多数 CUDA 时间 → 优化该 kernel。
- 若 CUDA 总时间 << 墙钟时间，或 aten::empty/cudaMalloc/launch 占比高 → 瓶颈是开销而非 kernel。
"""

import sys
import os
import argparse
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402

from framework.protocol import PROTOCOL, dtype_of  # noqa: E402


def load_case(name):
    return importlib.import_module(f"cases.{name}").CASE


def load_impl(spec):
    mod_path, fn_name = spec.split(":")
    return getattr(importlib.import_module(mod_path), fn_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--impl", default=None)
    ap.add_argument("--mode", choices=["forward", "backward", "both"], default="both")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("需要 CUDA GPU。请在 Colab 运行。")
        return 0

    case = load_case(args.case)
    impl = load_impl(args.impl or f"cases.{args.case}.op:candidate")
    dtype = dtype_of(case.dtype)
    dev = "cuda"

    inputs = case.make_inputs(0, dtype, dev, requires_grad=(args.mode != "forward"))
    with torch.no_grad():
        out_shape = case.reference_forward(
            case.make_inputs(0, dtype, dev, requires_grad=False), case.params).shape
    g = torch.Generator(device=dev).manual_seed(777)
    G = torch.randn(out_shape, dtype=dtype, device=dev, generator=g)

    def fwd_step():
        with torch.no_grad():
            impl({k: v.detach() for k, v in inputs.items()}, case.params)

    def full_step():
        loc = {k: (v.detach().clone().requires_grad_(True) if k in case.grad_inputs
                   else v.detach()) for k, v in inputs.items()}
        out = impl(loc, case.params)
        out.backward(G)

    step = fwd_step if args.mode == "forward" else full_step

    # warmup
    for _ in range(10):
        step()
    torch.cuda.synchronize()

    print(f"=== profile case={args.case} mode={args.mode} iters={args.iters} ===")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(args.iters):
            step()
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    return 0


if __name__ == "__main__":
    sys.exit(main())
