import torch, numpy as np
from faceage.utils.seed import set_seed

def test_seed_reproducible():
    set_seed(123); a = torch.randn(3)
    set_seed(123); b = torch.randn(3)
    assert torch.allclose(a,b)
    set_seed(123); n1 = np.random.randn(3)
    set_seed(123); n2 = np.random.randn(3)
    assert (n1==n2).all()