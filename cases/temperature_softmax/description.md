# Temperature Softmax

Turn each row of a batch of scores into a probability distribution whose
sharpness is controlled by a fixed positive temperature.

## Inputs and parameters

- `scores`: a contiguous float32 tensor with shape `[B, D]`. The default
  benchmark shape is `[4096, 1024]`.
- `temperature = 0.7`: a fixed positive scalar parameter. It is not learned
  and does not receive a gradient.
- `grad_inputs = ["scores"]`.

## Forward

For every row `b`, compute the numerically stable temperature-scaled softmax:

```text
z[b, d] = scores[b, d] / temperature
m[b] = max_d z[b, d]
probabilities[b, d]
    = exp(z[b, d] - m[b]) / sum_j exp(z[b, j] - m[b])
```

The output is float32 with shape `[B, D]`. Every output row is nonnegative and
sums to one. A temperature below one sharpens the distribution; a temperature
above one flattens it.

The operation normalizes each row independently along the final dimension. It
does not apply a mask and does not normalize across the batch dimension.
