import torch
from s5 import S5

x = torch.rand(2, 360, 24)
model = S5(24, 24)

y = model(x)
print(y.shape, y) # [2, 256, 32]
