import luz
import torch


def test_argmax():
    x = torch.tensor([0.0, 11.0, 3.0, 5, -0.2, 830, -10])

    assert torch.equal(luz.Argmax()(x), torch.tensor(5))
    assert torch.equal(luz.Argmax(dim=0)(x), torch.tensor(5))
    assert torch.equal(luz.Argmax(keepdim=True)(x), torch.tensor([5]))

    x = torch.tensor(
        [[0.0, 11.0, 3.0, 5, -0.2, 830, -10], [0.1, -30, 4000, 53.7, 2.34, -100, 100.0]]
    )

    assert torch.equal(luz.Argmax()(x), torch.tensor(9))
    assert torch.equal(luz.Argmax(dim=0)(x), torch.tensor([1, 0, 1, 1, 1, 0, 1]))
    assert torch.equal(luz.Argmax(dim=1)(x), torch.tensor([5, 2]))
    assert torch.equal(luz.Argmax(keepdim=True)(x), torch.tensor([[9]]))
    assert torch.equal(
        luz.Argmax(dim=0, keepdim=True)(x), torch.tensor([[1, 0, 1, 1, 1, 0, 1]])
    )
    assert torch.equal(luz.Argmax(dim=1, keepdim=True)(x), torch.tensor([[5], [2]]))


def test_identity():
    x1 = torch.tensor([0.0, 11.0, 3.0, 5, -0.2, 830, -10])
    x2 = torch.tensor(
        [[0.0, 11.0, 3.0, 5, -0.2, 830, -10], [0.1, -30, 4000, 53.7, 2.34, -100, 100.0]]
    )

    assert torch.equal(luz.Identity()(x1), x1)
    assert torch.equal(luz.Identity()(x2), x2)

    assert torch.equal(luz.Identity().inverse()(x1), x1)
    assert torch.equal(luz.Identity().inverse()(x2), x2)


def test_reshape():
    x = torch.tensor([0.0, 11.0, 3.0, -0.2, 830, -10])

    assert torch.equal(
        luz.Reshape(out_shape=(2, 3))(x),
        torch.tensor([[0.0, 11.0, 3.0], [-0.2, 830, -10]]),
    )
    assert torch.equal(
        luz.Reshape(out_shape=(1, 6))(x),
        torch.tensor([[0.0, 11.0, 3.0, -0.2, 830, -10]]),
    )
    assert torch.equal(
        luz.Reshape(out_shape=(3, -1))(x),
        torch.tensor([[0.0, 11.0], [3.0, -0.2], [830, -10]]),
    )
    assert torch.equal(
        luz.Reshape(out_shape=(1, 2, -1))(x),
        torch.tensor([[[0.0, 11.0, 3.0], [-0.2, 830, -10]]]),
    )


def test_squeeze():
    x1 = torch.tensor([[0.0, 11.0, 3.0, 5, -0.2, 830, -10]])
    x2 = torch.tensor(
        [
            [
                [0.0, 11.0, 3.0, 5, -0.2, 830, -10],
                [0.1, -30, 4000, 53.7, 2.34, -100, 100.0],
            ]
        ]
    )

    y1 = torch.tensor([0.0, 11.0, 3.0, 5, -0.2, 830, -10])
    y2 = torch.tensor(
        [
            [0.0, 11.0, 3.0, 5, -0.2, 830, -10],
            [0.1, -30, 4000, 53.7, 2.34, -100, 100.0],
        ]
    )

    assert torch.equal(luz.Squeeze(dim=0)(x1), y1)
    assert torch.equal(luz.Squeeze(dim=1)(x1), x1)
    assert torch.equal(luz.Squeeze(dim=0)(x2), y2)
    assert torch.equal(luz.Squeeze(dim=1)(x2), x2)
    assert torch.equal(luz.Squeeze(dim=2)(x2), x2)

    assert torch.equal(luz.Squeeze(dim=0).inverse()(y1), x1)
    assert torch.equal(luz.Squeeze(dim=1).inverse()(x1), x1.unsqueeze(dim=1))
    assert torch.equal(luz.Squeeze(dim=0).inverse()(y2), x2)
    assert torch.equal(luz.Squeeze(dim=1).inverse()(x2), x2.unsqueeze(dim=1))
    assert torch.equal(luz.Squeeze(dim=2).inverse()(x2), x2.unsqueeze(dim=2))


def test_unsqueeze():
    x1 = torch.tensor([0.0, 11.0, 3.0, 5, -0.2, 830, -10])
    x2 = torch.tensor(
        [[0.0, 11.0, 3.0, 5, -0.2, 830, -10], [0.1, -30, 4000, 53.7, 2.34, -100, 100.0]]
    )

    y1 = torch.tensor([[0.0, 11.0, 3.0, 5, -0.2, 830, -10]])
    y2 = torch.tensor([[0.0], [11.0], [3.0], [5], [-0.2], [830], [-10]])
    y3 = torch.tensor(
        [
            [
                [0.0, 11.0, 3.0, 5, -0.2, 830, -10],
                [0.1, -30, 4000, 53.7, 2.34, -100, 100.0],
            ]
        ]
    )
    y4 = torch.tensor(
        [
            [[0.0, 11.0, 3.0, 5, -0.2, 830, -10]],
            [[0.1, -30, 4000, 53.7, 2.34, -100, 100.0]],
        ]
    )
    y5 = torch.tensor(
        [
            [[0.0], [11.0], [3.0], [5], [-0.2], [830], [-10]],
            [[0.1], [-30], [4000], [53.7], [2.34], [-100], [100.0]],
        ]
    )

    assert torch.equal(luz.Unsqueeze(dim=0)(x1), y1)
    assert torch.equal(luz.Unsqueeze(dim=1)(x1), y2)
    assert torch.equal(luz.Unsqueeze(dim=0)(x2), y3)
    assert torch.equal(luz.Unsqueeze(dim=1)(x2), y4)
    assert torch.equal(luz.Unsqueeze(dim=2)(x2), y5)

    assert torch.equal(luz.Unsqueeze(dim=0).inverse()(y1), x1)
    assert torch.equal(luz.Unsqueeze(dim=1).inverse()(y2), x1)
    assert torch.equal(luz.Unsqueeze(dim=0).inverse()(y3), x2)
    assert torch.equal(luz.Unsqueeze(dim=1).inverse()(y4), x2)
    assert torch.equal(luz.Unsqueeze(dim=2).inverse()(y5), x2)


def test_scale():
    x = torch.tensor(
        [
            0.3582,
            0.9726,
            0.4059,
            0.5057,
            0.5394,
            0.9139,
            0.8946,
            0.0326,
            0.5348,
            0.5944,
            0.1723,
            0.6518,
            0.4743,
            0.2857,
            0.7778,
            0.1387,
            0.9085,
            0.5491,
            0.0680,
            0.3293,
            0.6467,
            0.1055,
            0.0590,
            0.2876,
            0.0448,
            0.3327,
            0.1708,
            0.3790,
            0.4383,
            0.7030,
            0.8725,
            0.9990,
            0.6376,
            0.8824,
            0.6957,
            0.7123,
            0.3367,
            0.5017,
            0.6979,
            0.3876,
            0.0497,
            0.0579,
            0.7030,
            0.3031,
            0.8677,
            0.3319,
            0.5606,
            0.0247,
            0.7529,
            0.6203,
            0.0645,
            0.3449,
            0.0201,
            0.1247,
            0.2931,
            0.2287,
            0.1541,
            0.4766,
            0.4624,
            0.4630,
            0.2633,
            0.1773,
            0.6150,
            0.4278,
            0.5446,
            0.4261,
            0.4274,
            0.5378,
            0.2204,
            0.2070,
            0.4256,
            0.0919,
            0.6791,
            0.4754,
            0.7079,
            0.2733,
            0.7480,
            0.2631,
            0.9129,
            0.7491,
            0.8927,
            0.6257,
            0.3653,
            0.2665,
            0.7630,
            0.3913,
            0.3229,
            0.4480,
            0.2298,
            0.4288,
            0.8689,
            0.6301,
            0.7956,
            0.5875,
            0.0344,
            0.9542,
            0.2771,
            0.1962,
            0.7008,
            0.4023,
        ]
    )

    scale = luz.Scale(x.mean(), x.std())

    z = scale(x)

    assert torch.allclose(z.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(z.std(), torch.tensor(1.0))
    assert torch.allclose(scale.inverse()(z), x)
