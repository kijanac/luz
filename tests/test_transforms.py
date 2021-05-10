import luz
import torch


def test_argmax():
    x = torch.tensor([0.0, 11.0, 3.0, 5, -0.2, 830, -10])

    assert torch.equal(luz.Argmax()(x), torch.tensor(5))
    assert torch.equal(luz.Argmax(dim=0)(x), torch.tensor(5))
    assert torch.equal(luz.Argmax(keepdim=True)(x), torch.tensor(5))

    x = torch.tensor(
        [[0.0, 11.0, 3.0, 5, -0.2, 830, -10], [0.1, -30, 4000, 53.7, 2.34, -100, 100.0]]
    )

    assert torch.equal(luz.Argmax()(x), torch.tensor(9))
    assert torch.equal(luz.Argmax(dim=0)(x), torch.tensor([1, 0, 1, 1, 1, 0, 1]))
    assert torch.equal(luz.Argmax(dim=1)(x), torch.tensor([5, 2]))
    assert torch.equal(luz.Argmax(keepdim=True)(x), torch.tensor(9))
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
