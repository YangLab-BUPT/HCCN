from sklearn.cluster import KMeans
import torch.nn as nn
from GNN_net import GraphSAGE
import torch.nn.functional as F
import torch
import pickle
import numpy as np
import pandas as pd
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from pronto import Ontology

device = torch.device('cuda:1')

# 加载 OBO 文件，解析 GO ID 到 namespace 的映射
go = Ontology("godata/go-basic-2311.obo")

model = GraphSAGE(768, 256, 128).to(device)
model.load_state_dict(torch.load('GNN_net.pth'))


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

go_id2ns = {
    go_id: go[go_id].namespace
    for go_id in valid_nodes
    if go_id in go
}

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
data = Data(x=x, edge_index=edge_index).to(device)


model.eval()
with torch.no_grad():
    z = model(data.x, data.edge_index)



# namespace 映射为整数标签
ns_label_map = {
    'molecular_function': 0,
    'biological_process': 1,
    'cellular_component': 2
}

# 给 valid_nodes 中每个节点打上 MFO/BPO/CCO 标签（如果缺失就跳过）
z_array = z.cpu().numpy()
z_2d = TSNE(n_components=2, perplexity=50).fit_transform(z_array)

ns_labels = []
z_2d_filtered = []

for i, go_id in enumerate(valid_nodes):
    ns = go_id2ns.get(go_id)
    if ns in ns_label_map:
        ns_labels.append(ns_label_map[ns])
        z_2d_filtered.append(z_2d[i])

z_2d_filtered = np.array(z_2d_filtered)
ns_labels = np.array(ns_labels)

# 可视化
plt.figure(figsize=(10, 8))
colors = ['#1f77b4', '#2ca02c', '#d62728']  # MFO / BPO / CCO
ns_names = ['MFO', 'BPO', 'CCO']

for i in range(3):
    idx = ns_labels == i
    plt.scatter(z_2d_filtered[idx, 0], z_2d_filtered[idx, 1], 
                c=colors[i], label=ns_names[i], s=10)

plt.legend()
plt.title("the t-SNE visualization of MFO / BPO / CCO in GNN")
plt.savefig('./image/tsne_GNN_namespace.png')
plt.show()
