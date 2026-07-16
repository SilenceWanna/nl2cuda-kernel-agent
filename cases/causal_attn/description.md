算法：单头 causal self-attention。

输入 Q、K、V 形状均为 [B,T,d]，fp32。batch 大小 B，序列长度 T，头维 d。

前向：
S[b,i,j] = dot(Q[b,i,:], K[b,j,:]) / sqrt(d)。
对 S 施加 causal mask：位置 i 只能看 j <= i，j > i 的上三角置为 -inf。
P = softmax(S, dim=-1)。
O[b,i,:] = sum_{j<=i} P[b,i,j] * V[b,j,:]，输出 O 形状 [B,T,d]。

反向：对 Q、K、V 都求梯度。
