"""通用计时 CLI：对指定 case 的候选实现 vs torch.compile 前反向计时。

用法：
    python skill/scripts/bench_case.py --case rbf
    python skill/scripts/bench_case.py --case rbf --impl cases.rbf.op:candidate

--case  <name>        : 加载 cases/<name>，取其 CASE 实例
--impl  <module:fn>   : 候选实现 "模块路径:函数名"；默认 cases/<name>/op.py:candidate
"""

import sys
import os
import re
import argparse
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

from framework.bench import compare  # noqa: E402


def load_case(name):
    return importlib.import_module(f"cases.{name}").CASE


def load_impl(spec):
    mod_path, fn_name = spec.split(":")
    return getattr(importlib.import_module(mod_path), fn_name)


def _parse_verdict(report, passed):
    """从 compare() 的报告文本解析机读摘要。

    passed（来自 framework/bench.compare()，权威）决定 PASS vs BENCH_FAIL；
    报告中的 "存在 CV>5%" 标记 → cv_ok=0 → CV_INVALID（让调用方重测而非信噪声）。
    加速比锚在 "x  (需" 上，避免误抓 "... ms" 计时行。
    返回单行字符串，形如 "VERDICT=PASS fwd=1.09x bwd=1.40x cv_ok=1"。
    """
    # 前向/反向加速比行： "  forward : 1.0921x  (需 ≥1.05x) PASS"
    speedups = re.findall(r"(forward|backward)\s*:\s*([0-9.]+)x\s*\(需", report)
    sp = {k: v for k, v in speedups}
    fwd = sp.get("forward", "?")
    bwd = sp.get("backward", "?")
    cv_ok = 0 if "存在 CV>5%" in report else 1

    if cv_ok == 0:
        verdict = "CV_INVALID"
    elif passed:
        verdict = "PASS"
    else:
        verdict = "BENCH_FAIL"
    return f"VERDICT={verdict} fwd={fwd}x bwd={bwd}x cv_ok={cv_ok}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="case 名，如 rbf")
    ap.add_argument("--impl", default=None, help="候选实现 module:fn；默认 cases.<case>.op:candidate")
    ap.add_argument("--strict", action="store_true",
                    help="达标 exit 0、未达标 exit 1（供自动闭环用退出码判定）")
    ap.add_argument("--emit-verdict", action="store_true",
                    help="报告后额外打印单行机读摘要 VERDICT=...")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("需要 CUDA GPU。本地无 GPU，请在 Colab 运行。")
        return 0

    case = load_case(args.case)
    impl_spec = args.impl or f"cases.{args.case}.op:candidate"
    impl = load_impl(impl_spec)

    print(f"case={args.case}  impl={impl_spec}")
    report, passed = compare(case, impl, device="cuda")
    print(report)

    if args.emit_verdict:
        print(_parse_verdict(report, passed))

    # 默认行为不变（无 flag 恒 exit 0，仅报告）；--strict 才用退出码表达达标与否。
    if args.strict:
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
