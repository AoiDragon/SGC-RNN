import scipy.stats as stats
import torch

# a = torch.ones(1, 3)
# b = torch.ones(1, 3)
# print(a)
#
# print(torch.cat((a, b), 1))
# print(torch.cat((a, b), 0))
#
# print(torch.full((1, 10), -1))

a = []
a.append(torch.ones(2))

a[0][1] = -1
print(a)