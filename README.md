# S5: Simplified State Space Layers for Sequence Modeling
This is a ported version derived from <https://github.com/lindermanlab/S5> and <https://github.com/kavorite/S5>.
It includes a bunch of functions ported from jax/lax/flax/whatever since they didn't exist yet. 

Jax is required because it relies on the pytree structure but it's not used for any computation. 
Pytorch 1.13 or later is required because it makes heavy use of `functorch.vamp` to substitute it's jax counterpart.
Python 3.10 or later is required due to usage of the `match` keyword

\--- 

Update:

In my experiments it follows the results found in the [Hyena Hierarchy](https://arxiv.org/abs/2302.10866) (& H3) paper that the state spaces alone lack the recall capabilities required for LLM but seem work well for regular sequence feature extraction and linear complexity.

You can use variable step-size as described in the paper using a 1D tensor for `step_scale` however this takes **a lot of memory** due to a lot of intermediate values needing to be held (which I believe is true for the official S5 repo, but not mentioned in the paper unless I missed it).

## Install

```sh
pip install s5-pytorch 
```

## Example

```py3
from s5 import S5, S5Block

# Raw S5 operator
x = torch.rand([2, 256, 32])
model = S5(32, 32)
model(x) # [2, 256, 32]

# S5-former block (S5+FFN-GLU w/ layernorm, dropout & residual)
model = S5Block(32, 32, False)
model(x) # [2, 256, 32]
```
