import torch.profiler as profiler
import torch
import torchinfo
from s5 import S5, S5Block

x = torch.rand(2, 8192, 256)
# model = S5(32, 32)
model = S5Block(256, 512, block_count=8, bidir=False)

print(torchinfo.summary(model, (2, 8192, 256), device='cpu'))

y = model(x)
print(y.shape, y) # [2, 256, 32]

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    res = model(x)
    
print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_memory_usage", row_limit=10))