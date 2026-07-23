# Segment softmax (scatter softmax)

Inputs:

- `values`: float32 tensor with shape `[N, ...]`.
- `segment_ids`: int64 tensor with shape `[N]`; every value is in `[0, num_segments - 1]`.
- `num_segments`: positive integer scalar.

The operator normalizes independently inside every segment and every trailing
feature position.  For `v = values[i, ...]`, `s = segment_ids[i]`, and a
trailing feature position `f`, the output is

`output[i, f] = exp(values[i, f] - max(values[j, f] for j if segment_ids[j] == s)) / sum(exp(values[k, f] - max(values[j, f] for j if segment_ids[j] == s)) for k if segment_ids[k] == s)`.

Empty segments have no corresponding output elements.  Gradients are required
only for `values`; `segment_ids` is an integer index tensor.  With upstream
gradient `G`, the backward relation is

`dvalues[i, f] = output[i, f] * (G[i, f] - sum(G[j, f] * output[j, f] for j if segment_ids[j] == segment_ids[i]))`.
