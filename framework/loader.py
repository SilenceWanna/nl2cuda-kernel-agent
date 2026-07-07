"""CUDA 扩展即时编译加载器（阶段1+复用）。

用 torch.utils.cpp_extension.load 在运行环境（Colab T4）即时编译 .cu 源文件，
返回可调用的 python 扩展模块。编译参数针对 sm_75（Turing / T4）。

精度约定（重要）：**默认不开 --use_fast_math**。
- fast math 会降低 exp/除法精度，既可能使正确性 allclose 翻车，也触碰防作弊
  "禁止降精度换速度" 红线。先保证正确；若优化阶段需要，再单独、显式评估。
"""

import os

import torch
from torch.utils.cpp_extension import load


FRAMEWORK_DIR = os.path.dirname(os.path.abspath(__file__))

# 针对 T4(sm_75) 的默认编译参数。-O3 常规优化；不含 fast-math。
DEFAULT_CUDA_FLAGS = [
    "-O3",
    "-gencode", "arch=compute_75,code=sm_75",
]
DEFAULT_CPP_FLAGS = ["-O3"]


def load_kernel(name, sources, base_dir=None, extra_cuda_cflags=None,
                extra_cflags=None, verbose=True):
    """即时编译并加载一个 CUDA 扩展。

    name: 扩展模块名（须唯一，用于构建缓存目录）。
    sources: .cu / .cpp 源文件名列表（绝对路径，或相对 base_dir）。
    base_dir: 相对源文件的基准目录；默认 framework/（用于 smoke.cu）。
              各 case 应传入自己的 kernels 目录，如 cases/rbf/kernels。
    extra_cuda_cflags: 追加到默认 CUDA 编译参数之后的额外参数。
    返回：已加载的扩展模块对象。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能编译/加载 kernel（本地无 GPU，请在 Colab 运行）")

    base = base_dir if base_dir is not None else FRAMEWORK_DIR
    resolved = []
    for s in sources:
        p = s if os.path.isabs(s) else os.path.join(base, s)
        if not os.path.exists(p):
            raise FileNotFoundError(f"源文件不存在: {p}")
        resolved.append(p)

    cuda_flags = list(DEFAULT_CUDA_FLAGS)
    if extra_cuda_cflags:
        cuda_flags += list(extra_cuda_cflags)
    cpp_flags = list(DEFAULT_CPP_FLAGS)
    if extra_cflags:
        cpp_flags += list(extra_cflags)

    return load(
        name=name,
        sources=resolved,
        extra_cuda_cflags=cuda_flags,
        extra_cflags=cpp_flags,
        verbose=verbose,
    )
