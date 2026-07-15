Algorithm: linear state space model (linear SSM).

Forward:
For input X with shape [B, T, C] in fp32, scan along the time dimension:

    h_t[b, c] = a * h_{t-1}[b, c] + b_coef * X[b, t, c]
    h_{-1}[b, c] = 0
    Y[b, t, c] = h_t[b, c]

Constants:
- a = 0.9
- b_coef = 1.0

The constants are not learnable parameters. Compute gradients only for X.
