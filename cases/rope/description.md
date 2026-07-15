# RoPE (Rotary Position Embedding)

Input `X` has shape `[B, S, H, D]`, dtype `float32`, and positive even `D`.
For each adjacent pair `(x_{2i}, x_{2i+1})`, where `i = 0, ..., D/2-1`,
the angle at sequence position `s` is

`theta(s, i) = s * base^(-2i/D)`, with `base = 10000`.

The forward output `Y`, with the same shape as `X`, is

`y_{2i}   = x_{2i} * cos(theta) - x_{2i+1} * sin(theta)`

`y_{2i+1} = x_{2i} * sin(theta) + x_{2i+1} * cos(theta)`.

Compute the gradient with respect to `X`.
