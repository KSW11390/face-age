import torch
from faceage.models.cnn import SimpleCNN
from faceage.models.head import SoftHead

def test_forward_shapes():
    x = torch.randn(4, 3, 200, 200)
    feat = SimpleCNN()(x)
    assert feat.shape == (4,128)
    race = torch.zeros(4,5)
    race[:,0]=1
    logits = SoftHead(128, num_bins=86)(feat, race)
    assert logits.shape == (4,86)