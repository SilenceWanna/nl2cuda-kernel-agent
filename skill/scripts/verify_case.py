"""通用正确性验证 CLI：对指定 case 的候选实现对拍参考金标准。

用法：
    python skill/scripts/verify_case.py --case rbf
    python skill/scripts/verify_case.py --case rbf --impl cases.rbf.op:candidate

--case  <name>        : 加载 cases/<name>，取其 CASE 实例
--impl  <module:fn>   : 候选实现的 "模块路径:函数名"；默认 cases/<name>/op.py:candidate
"""

import sys
import os
import argparse
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

from framework.verify import verify_all  # noqa: E402


def load_case(name):
    mod = importlib.import_module(f"cases.{name}")
    return mod.CASE


def load_impl(spec):
    mod_path, fn_name = spec.split(":")
    mod = importlib.import_module(mod_path)
    return getattr(mod, fn_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="case 名，如 rbf")
    ap.add_argument("--impl", default=None, help="候选实现 module:fn；默认 cases.<case>.op:candidate")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("需要 CUDA GPU。本地无 GPU，请在 Colab 运行。")
        return 0

    case = load_case(args.case)
    impl_spec = args.impl or f"cases.{args.case}.op:candidate"
    impl = load_impl(impl_spec)

    print(f"case={args.case}  impl={impl_spec}")
    ok, report = verify_all(case, impl, device="cuda")
    print(report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
