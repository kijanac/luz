import luz
import torch

d_v = 17
d_e = 13
d_u = 5
d_attn = 40

nodes = torch.rand((10, d_v))
edge_index = torch.tensor(
    [[0, 2], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 2], [7, 8], [8, 9]]
).T
edge_index = torch.cat((edge_index, edge_index.flipud()), dim=1)
_, N_e = edge_index.shape
edges = torch.rand((N_e, d_e))
u = torch.rand((2, d_u))
batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

# attn = luz.MultiheadEdgeAttention(7, d_v, d_e, d_u, d_attn, nodewise=True)
# print(attn(nodes, edges, edge_index, u, batch).shape)

attn = luz.EdgeAggregateLocal(d_v, d_e, d_u, d_attn, 7)
print(attn(nodes, edges, edge_index, u, batch).shape)

# attn = luz.AdditiveNodeAttention(d_v,d_attn)
# print(attn(nodes, edge_index))

# attn = luz.MultiheadNodeAttention(7, d_v, d_u)
# print(attn(nodes, edges, edge_index, u, batch).shape)

# gat = luz.EdgeAttention(d_v,d_e,d_u,d_attn)

# out = gat(nodes,edges,edge_index,u,batch)

# gcn = luz.GraphConv(d_v,torch.nn.functional.leaky_relu)
# #print(gcn(nodes,edge_index))
# #gnt = luz.NodeAttention(d_v, torch.nn.functional.leaky_relu)
# ##print(gnt(nodes,edge_index,u,batch))

# gca = luz.GraphConvAttention(d_v,torch.nn.functional.leaky_relu)
# #print(gca(nodes,edge_index,batch))

# gaa = luz.AdditiveAttention(d_v, d_attn)
# print(gaa(nodes, edge_index))
