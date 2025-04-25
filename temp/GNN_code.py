import torch
from torch_geometric.data import Data
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import torch.nn.functional as F
from GNN_net import GraphSAGE

# 加载 BERT 向量和 ID
bert_emb = np.load('godata/go_sapbert_cls_embeddings.npy')  # (N, 768)
bert_ids = list(pd.read_csv('godata/go_sapbert_ids.csv')['GO_ID'])  # 长度 N

# 构建 GO ID -> embedding 的映射
id2emb = {go_id: emb for go_id, emb in zip(bert_ids, bert_emb)}

with open("godata/go_subgraph.pkl", "rb") as f:
    G = pickle.load(f)

# 只使用图中存在的节点，且在bert_emb中也有的
node_list = list(G.nodes())
valid_nodes = [n for n in node_list if n in id2emb]

# 最终用这个作为 GNN 输入的节点列表和向量
x = torch.tensor([id2emb[n] for n in valid_nodes], dtype=torch.float)

# 构建 node2idx 映射（顺序必须和 x 一致）
node2idx = {node: idx for idx, node in enumerate(valid_nodes)}

edges = []
for u, v in G.edges():
    if u in node2idx and v in node2idx:
        edges.append((node2idx[u], node2idx[v]))

# 转为 PyG 的 edge_index
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# 构建图数据
data = Data(x=x, edge_index=edge_index)



model = GraphSAGE(768, 256, 128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
epochs = 200

model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)  # 输出嵌入 (num_nodes, 128)

    # 正样本
    src, dst = data.edge_index
    pos_sim = F.cosine_similarity(z[src], z[dst]).mean()

    # 负样本
    neg_dst = dst[torch.randperm(dst.size(0))]
    neg_sim = F.cosine_similarity(z[src], z[neg_dst]).mean()

    loss = -(pos_sim - neg_sim)  # 希望正样本更相似，负样本不相似
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


torch.save(model.state_dict(), 'GNN_net.pth')