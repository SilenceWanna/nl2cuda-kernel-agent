"""环境探测脚本：确认 Colab 的 GPU 型号、CUDA / PyTorch 版本、nvcc 可用性。

用法（在 Colab 或任意带 GPU 的机器）：
    python scripts/probe_env.py

输出决定后续 CUDA 编译参数（如是否支持 TF32、目标 sm 架构）。
"""

import subprocess
import sys


def run(cmd):
    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60
        )
        return (out.stdout + out.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"<failed: {e}>"


def main():
    print("=" * 60)
    print("Python:", sys.version.replace("\n", " "))

    print("=" * 60)
    print("nvidia-smi:")
    print(run("nvidia-smi"))

    print("=" * 60)
    print("nvcc --version:")
    print(run("nvcc --version"))

    print("=" * 60)
    try:
        import torch

        print("torch:", torch.__version__)
        print("torch CUDA build:", torch.version.cuda)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(i)
            cc = (props.major, props.minor)
            print(f"device[{i}]:", props.name)
            print("compute capability:", f"{cc[0]}.{cc[1]}  (sm_{cc[0]}{cc[1]})")
            print("total memory (GB):", round(props.total_memory / 1024**3, 2))
            print("multiprocessors (SM count):", props.multi_processor_count)
            # TF32 需要 Ampere (sm_80) 及以上
            tf32_ok = cc[0] >= 8
            print("TF32 supported (needs sm_80+):", tf32_ok)
            # FP16 tensor core 需要 sm_70+ (Volta/Turing)
            print("FP16 tensor core (needs sm_70+):", cc[0] >= 7)
    except Exception as e:  # noqa: BLE001
        print("torch import/probe failed:", e)

    print("=" * 60)
    try:
        import numpy

        print("numpy:", numpy.__version__)
    except Exception as e:  # noqa: BLE001
        print("numpy not available:", e)


if __name__ == "__main__":
    main()
