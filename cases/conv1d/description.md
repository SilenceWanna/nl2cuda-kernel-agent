算法：depthwise causal conv1d（逐通道因果一维卷积）。

前向：输入 X 形状 [B,C,T]（fp32，batch B、通道 C、时序长 T），卷积核 W 形状 [C,K]，每通道独立，K=4。输出 Y 形状 [B,C,T]：

y[b,c,t] = sum_{k=0}^{K-1} w[c,k] * x[b,c,t-k]

当 t-k < 0 时 x 视为 0（因果左 padding）。每个通道 c 使用自己的核 w[c,:]，通道间不混合。对 X 和 W 都求梯度。
