import luz
import torch


def test_adjacency():
    edge_index = torch.tensor(
        [[0, 2], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 2]]
    ).T
    A = torch.tensor(
        [
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0],
        ]
    )

    assert torch.allclose(A, luz.adjacency(edge_index))

    edge_index = torch.cat((edge_index, edge_index.flipud()), dim=1)
    A = torch.tensor(
        [
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 0],
        ]
    )

    assert torch.allclose(A, luz.adjacency(edge_index))


def test_batchwise_mask():
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
    m = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    assert torch.allclose(m, luz.batchwise_mask(batch))

    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    m = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    assert torch.allclose(m, luz.batchwise_mask(batch))

    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
    edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 2], [1, 2, 3, 4, 5, 6, 8]])
    m = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    assert torch.allclose(m, luz.batchwise_mask(batch, edge_index))

    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 2], [1, 2, 3, 4, 5, 6, 8]])
    m = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    assert torch.allclose(m, luz.batchwise_mask(batch, edge_index))

    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    edge_index = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1, 2, 9, 10, 11, 12, 13],
            [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14],
        ]
    )
    m = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    assert torch.allclose(m, luz.batchwise_mask(batch, edge_index))


def test_batchwise_edge_sum():
    edges = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
        ]
    )
    edge_index = torch.tensor(
        [[0, 1], [0, 2], [0, 3], [0, 4], [1, 5], [1, 6], [2, 8]]
    ).T
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
    s = torch.tensor([[28.0, 28.0]])
    assert torch.allclose(s, luz.batchwise_edge_sum(edges, edge_index, batch))

    edges = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
        ]
    )
    edge_index = torch.tensor(
        [[0, 1], [0, 2], [0, 3], [0, 4], [1, 5], [1, 6], [2, 8]]
    ).T
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    s = torch.tensor([[28.0, 28.0]])
    assert torch.allclose(s, luz.batchwise_edge_sum(edges, edge_index, batch))

    edges = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
            [8.0, 8.0],
            [9.0, 9.0],
            [10.0, 10.0],
            [11.0, 11.0],
            [12.0, 12.0],
        ]
    )
    edge_index = torch.tensor(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 5],
            [1, 6],
            [2, 8],
            [9, 10],
            [10, 11],
            [11, 12],
            [12, 13],
            [13, 14],
        ]
    ).T
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    s = torch.tensor([[28.0, 28.0], [50.0, 50.0]])
    assert torch.allclose(s, luz.batchwise_edge_sum(edges, edge_index, batch))


def test_batchwise_edge_mean():
    edges = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
        ]
    )
    edge_index = torch.tensor(
        [[0, 1], [0, 2], [0, 3], [0, 4], [1, 5], [1, 6], [2, 8]]
    ).T
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
    s = torch.tensor([[4.0, 4.0]])
    assert torch.allclose(s, luz.batchwise_edge_mean(edges, edge_index, batch))

    edges = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
        ]
    )
    edge_index = torch.tensor(
        [[0, 1], [0, 2], [0, 3], [0, 4], [1, 5], [1, 6], [2, 8]]
    ).T
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    s = torch.tensor([[4.0, 4.0]])
    assert torch.allclose(s, luz.batchwise_edge_mean(edges, edge_index, batch))

    edges = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
            [8.0, 8.0],
            [9.0, 9.0],
            [10.0, 10.0],
            [11.0, 11.0],
            [12.0, 12.0],
        ]
    )
    edge_index = torch.tensor(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 5],
            [1, 6],
            [2, 8],
            [9, 10],
            [10, 11],
            [11, 12],
            [12, 13],
            [13, 14],
        ]
    ).T
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    s = torch.tensor([[4.0, 4.0], [10.0, 10.0]])
    assert torch.allclose(s, luz.batchwise_edge_mean(edges, edge_index, batch))


def test_in_degree():
    A = torch.tensor(
        [
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0],
        ]
    )

    D_in = torch.tensor([0, 0, 3, 1, 1, 1, 1])

    assert torch.allclose(D_in, luz.in_degree(A))

    A = torch.tensor(
        [
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 0],
        ]
    )

    D_in = torch.tensor([1, 1, 4, 2, 2, 2, 2])

    assert torch.allclose(D_in, luz.in_degree(A))


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


def test_out_degree():
    A = torch.tensor(
        [
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0],
        ]
    )

    D_out = torch.tensor([1, 1, 1, 1, 1, 1, 1])

    assert torch.allclose(D_out, luz.out_degree(A))

    A = torch.tensor(
        [
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 0],
        ]
    )

    D_out = torch.tensor([1, 1, 4, 2, 2, 2, 2])

    assert torch.allclose(D_out, luz.out_degree(A))
