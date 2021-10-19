import torch
import numpy as np
import torch.nn.functional as F


h_tmp = []
h_tmp.append(torch.randn(7, 1))
h_tmp.append(torch.randn(7, 1))
h_tmp.append(torch.randn(7, 1))
h_0 = torch.randn(7, 1)
for h_now in h_tmp:
    h_0 = torch.cat((h_now, h_0), 0)

h_1 = F.normalize(h_0, p=2, dim=0)
print(h_1)
