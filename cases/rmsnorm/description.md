算法：RMSNorm。

前向：
- 输入：
  - X: shape [4096, 1024]，fp32
  - gamma: shape [1024]，fp32
- 标量参数：
  - eps = 1e-5
- 计算：
  - rms = sqrt(mean(X^2, dim=-1, keepdim=True) + eps)
  - y = X / rms * gamma

反向：
- 对 X、gamma 求梯度。

实现要求：
- 使用自定义 CUDA 前向与反向 kernel。
- 精度保持 fp32。
- 不使用 torch.nn.functional 等高层融合算子。
