"""Case 协议：framework 与算法 case 之间的唯一接口。

一个 case 封装某个算法的全部特定信息；framework 的 verify/bench 只依赖此协议，
不假设输入个数、输出形状或参数名——这是支持任意算法结构的关键。

新增一个算法 = 在 cases/<name>/ 下实现一个 Case 实例并在 __init__ 暴露 CASE。
"""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Case:
    """算法 case 描述。

    字段：
      name: 短名，如 "rbf"。
      description: 自然语言算法描述（skill 的输入样例）。
      params: 算法标量参数字典，如 {"gamma": 1/64}。
      grad_inputs: 需要求梯度的输入名列表，如 ["X", "Y"]。
      dtype: 验收精度，如 "float32"。
      make_inputs: (seed, dtype, device, requires_grad) -> dict[str, Tensor]
                   返回命名输入张量，键名即梯度归属。
      reference_forward: (inputs: dict, params: dict) -> Tensor
                   PyTorch 金标准前向；autograd 提供反向。
      grad_shapes: 可选，(seed,dtype,device)->输出张量形状生成用；默认由前向输出推断。
    """
    name: str
    description: str
    params: dict
    grad_inputs: list
    dtype: str
    make_inputs: Callable
    reference_forward: Callable
