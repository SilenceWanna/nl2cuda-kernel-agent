# GroupNorm

Input `X` is a contiguous float32 NCHW feature map with shape `[N,C,H,W]`.
Split its channels into `G` equal contiguous groups. For every sample and group,
normalize all `(C/G)*H*W` values together using the population variance:

```
Z = reshape(X, [N,G,C/G,H,W])
mean = Z.mean((2,3,4), keepdim=True)
var = ((Z - mean) ** 2).mean((2,3,4), keepdim=True)
Xhat = reshape((Z - mean) / sqrt(var + eps), [N,C,H,W])
Y[n,c,h,w] = Xhat[n,c,h,w] * gamma[c] + beta[c]
```

`gamma` and `beta` are float32 vectors with shape `[C]`. The output has the
same shape and dtype as `X`. Gradients are required for `X`, `gamma`, and
`beta`. There are no running statistics. Defaults are `N=64`, `C=128`,
`H=W=56`, `G=32`, and `eps=1e-5`; shape values support environment overrides.
