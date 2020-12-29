import luz
import torch


def test_masked_softmax():
    mask = torch.tensor(
        [
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 0, 1],
        ]
    )

    x = torch.tensor(
        [
            [0.1948, 0.1944, 0.4431, 0.2659, 0.7232],
            [0.6129, 0.5780, 0.9547, 0.5576, 0.6854],
            [0.3941, 0.9102, 0.1644, 0.1464, 0.3381],
            [0.9598, 0.1323, 0.9322, 0.9481, 0.3282],
            [0.5216, 0.0182, 0.7681, 0.3139, 0.3193],
        ]
    )

    assert torch.allclose(
        luz.masked_softmax(x, mask, dim=0).sum(dim=0),
        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    assert torch.allclose(
        luz.masked_softmax(x, mask, dim=1).sum(dim=1),
        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
    )
