# S5: Simplified State Space Layers for Sequence Modeling
This is a ported version derived from <https://github.com/lindermanlab/S5> and <https://github.com/kavorite/S5>.
It includes a bunch of functions ported from jax/lax/flax/whatever since they didn't exist yet. 

Jax is required because it relies on the pytree structure but it's not used for any computation. 
Pytorch 1.13 or later is required because it makes heavy use of `functorch.vamp` to substitute it's jax counterpart.
Python 3.10 or later is required due to usage of the `match` keyword

## Install
```
pip install s5-pytorch 
```

## Example

```
from s5 import S5

x = torch.rand([2, 256, 32])
model = S5(32, 32)

model(x) # [2, 256, 32]
```