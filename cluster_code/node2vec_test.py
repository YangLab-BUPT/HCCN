import torch
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from torch_geometric.nn import Node2Vec

# 加载 NetworkX 图
with open("godata/go_subgraph.pkl", "rb") as f:
    G = pickle.load(f)

# 创建 GO term ID 到整数索引的映射
node2idx = {node: idx for idx, node in enumerate(G.nodes())}
idx2node = {idx: node for node, idx in node2idx.items()}

# 构建 edge_index
edges = [(node2idx[u], node2idx[v]) for u, v in G.edges()]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# 初始化并训练 Node2Vec 模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Node2Vec(edge_index, embedding_dim=128, walk_length=30, context_size=10,
                 walks_per_node=10, num_negative_samples=1, sparse=True).to(device)
loader = model.loader(batch_size=128, shuffle=True)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

for epoch in range(1, 10):
    loss = train()
    print(f'Epoch: {epoch}, Loss: {loss:.4f}')

# 获取嵌入向量
embeddings = model.embedding.weight.data.cpu().numpy()

# 创建 DataFrame，将 GO term ID 与嵌入向量对应
go_ids = [idx2node[i] for i in range(len(embeddings))]
df = pd.DataFrame(embeddings, index=go_ids)
df.index.name = 'GO_ID'

# 保存为 CSV 文件
df.to_csv("./godata/go_node2vec_embeddings.csv")
