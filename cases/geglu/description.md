# Tanh-GeGLU

Given contiguous fp32 input `X[B,T,2H]`, split the last dimension into
`V = X[..., :H]` and `G = X[..., H:]`. The output is `Y[B,T,H]`:

\[
u = \sqrt{\frac{2}{\pi}}\left(G + 0.044715G^3\right),
\qquad
\operatorname{GELU}_{\tanh}(G) = \frac{1}{2}G(1 + \tanh(u)),
\qquad
Y = V \odot \operatorname{GELU}_{\tanh}(G).
\]

The backward pass returns the gradient for all of `X`:

\[
dV = dY \odot \operatorname{GELU}_{\tanh}(G),
\]

\[
dG = dY \odot V \odot \left[
\frac{1}{2}(1+\tanh(u)) +
\frac{1}{2}G(1-\tanh^2(u))\sqrt{\frac{2}{\pi}}
(1+3\cdot0.044715G^2)
\right].
\]

The default benchmark shape is `B=16`, `T=2048`, `H=4096`. Override it with
`GEGLU_B`, `GEGLU_T`, and `GEGLU_H`.
