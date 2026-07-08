# RBF 纯 CUDA 交付版 —— 独立编译（仅需 nvcc + CUDA runtime，不依赖 PyTorch）
#
#   make            编译静态库 librbf.a 与自测程序 rbf_test
#   make test       编译并运行自测（CPU 参考对拍，打印 PASS/FAIL）
#   make clean      清理
#
# 目标架构默认 sm_80（A100）。改其他卡：make ARCH=sm_75（T4）等。

NVCC   ?= nvcc
ARCH   ?= sm_80
NVFLAGS = -O3 -arch=$(ARCH) -std=c++17

all: rbf_test librbf.a

# 交付 kernel 目标文件
rbf_kernels.o: rbf_kernels.cu
	$(NVCC) $(NVFLAGS) -c rbf_kernels.cu -o rbf_kernels.o

# 静态库（供外部 C/C++ 项目链接；接口见 rbf_kernels.cu 顶部 extern "C" 声明）
librbf.a: rbf_kernels.o
	ar rcs librbf.a rbf_kernels.o

# 自测程序
rbf_test: rbf_kernels.cu rbf_test.cu
	$(NVCC) $(NVFLAGS) rbf_kernels.cu rbf_test.cu -o rbf_test

test: rbf_test
	./rbf_test

clean:
	rm -f rbf_test rbf_kernels.o librbf.a

.PHONY: all test clean
