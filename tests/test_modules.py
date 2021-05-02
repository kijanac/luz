# import luz
# import torch


# # def test_diff_pool():
# d_v = 10
# d_e = 13
# d_u = 5
# nodes = torch.rand((10, d_v))
# edge_index = torch.tensor(
#     [[0, 2], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 2], [7, 8], [8, 9]]
# ).T
# edge_index = torch.cat((edge_index, edge_index.flipud()), dim=1)
# _, N_e = edge_index.shape
# edges = torch.rand((N_e, d_e))
# u = torch.rand((2, d_u))
# batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
# d = luz.Data(
#     x=nodes,
#     edge_attr=edges,
#     edge_index=edge_index,
#     u=u,
#     y=torch.tensor([1.0]),
#     batch=batch,
# )
# gcn = luz.GraphConv(d_v, torch.nn.SELU())
# # print(gcn(d.x, d.edge_index)[:,:5]*luz.batchwise_mask(batch))
# # s = luz.masked_softmax(d.x,luz.batchwise_mask(batch),dim=0)
# num_clusters = 3

# agp = luz.AverageGraphPool(num_clusters)
# s = gcn(d.x, d.edge_index)[:, :num_clusters]

# agp(d.x, d.edge_attr, d.edge_index, d.batch, s)

# # test_diff_pool()
