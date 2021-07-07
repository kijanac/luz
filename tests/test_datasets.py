import luz
import torch


def test_mean_std():
    d = luz.Dataset([luz.Data(x=x) for x in torch.linspace(0, 100, 100)])

    mean, std = d.mean_std(key="x")

    assert torch.isclose(mean, torch.tensor(50.0))
    assert torch.isclose(std, torch.tensor(29.157646512850633))
