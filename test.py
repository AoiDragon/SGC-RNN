import scipy.stats as stats
import torch

mu, sigma = 0.5, 1
lower, upper = 0, 1
X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)  # 生成分布
h = X.rvs(7)  # 取样
h = torch.from_numpy(h)
print(h)