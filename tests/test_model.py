import torch

from tiger_reid.model import EmbeddingNet


def test_embedding_shape_and_unit_norm():
    m = EmbeddingNet("resnet18", dim=64, pretrained=False).eval()
    with torch.no_grad():
        z = m(torch.randn(3, 3, 64, 64))
    assert z.shape == (3, 64)
    assert torch.isfinite(z).all()
    assert torch.allclose(z.norm(dim=1), torch.ones(3), atol=1e-5)


def test_param_groups_cover_all_params():
    m = EmbeddingNet("resnet18", dim=32, pretrained=False)
    groups = m.param_groups(1e-4, 1e-3)
    n = sum(p.numel() for g in groups for p in g["params"])
    assert n == sum(p.numel() for p in m.parameters())
