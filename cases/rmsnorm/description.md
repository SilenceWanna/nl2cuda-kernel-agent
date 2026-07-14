算法：RMSNorm。

输入：
- X: shape [4096, 1024]，float32
- gamma: shape [1024]，float32

前向：
1. 对每一行计算
   rms = sqrt(mean(x^2) + eps)
2. 输出
   y = x / rms * gamma

其中 eps = 1e-5。

梯度：
- 对 X 求梯度
- 对 gamma 求梯度
