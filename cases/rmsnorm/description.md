RMSNorm。

前向定义：
- 输入：
  - X: shape [4096, 1024], dtype fp32
  - gamma: shape [1024], dtype fp32
- 标量参数：
  - eps = 1e-5
- 计算：
  - rms = sqrt(mean(X^2, dim=-1, keepdim=True) + eps)
  - Y = X / rms * gamma

梯度需求：
- 对 X 求梯度
- 对 gamma 求梯度
