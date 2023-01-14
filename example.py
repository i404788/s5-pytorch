import torch
from s5 import S5

x = torch.rand([2, 256, 32])
model = S5(32, 32)

y = model(x)
print(y.shape, y) # [2, 256, 32]
