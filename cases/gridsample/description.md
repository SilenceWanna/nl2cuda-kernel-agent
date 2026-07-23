# Bilinear grid sample

The input is a contiguous `float32` tensor `X[N, C, H, W]` and `grid[N, OH, OW, 2]`.
The last grid dimension stores normalized `(x, y)` coordinates.  For each output
location, the CUDA kernel maps the coordinate to input pixel space, samples the
four surrounding pixels with bilinear weights, and uses zero padding for
neighbors outside the input image.  Coordinate conversion always uses
`align_corners=False`.  The output has shape `Y[N, C, OH, OW]`.

Only `X` is differentiable.  The backward pass scatters each upstream gradient
to the four input neighbors with the forward interpolation weights and
`atomicAdd`.  `grid` is a constant and never receives a gradient.  NaN grid
coordinates follow PyTorch's grid-sample convention and are treated as `-1`.
