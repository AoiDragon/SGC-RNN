import torch
import numpy as np
import torch.nn.functional as F


# input_size, hidden_size
rnn = torch.nn.LSTM(7, 10)
# sequence length, batch_size, h_size
input = torch.randn(5, 2, 7)
h0 = torch.randn(1, 2, 10)
c0 = torch.randn(1, 2, 10)
output, (hn, cn) = rnn(input, (h0, c0))
# 输出有三个维度，第一个是序列长度，第二个是batch_size，第三个是输出的大小
# 可以通过切片，分割每一个batch的输出
# 默认的序列长度是5，如果小于5就趣对应轮数的结果送入下一层
print(output.shape)
print(hn.shape)
print(cn.shape)
