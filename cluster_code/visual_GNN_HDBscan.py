import hdbscan
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
from sklearn.decomposition import PCA


model = GraphSAGE(768, 256, 128)
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

# 最终用这个作为 GNN 输入的节点列表和向量
x = torch.tensor([id2emb[n] for n in valid_nodes], dtype=torch.float)

# 构建 node2idx 映射（顺序必须和 x 一致）
node2idx = {node: idx for idx, node in enumerate(valid_nodes)}

edges = []
for u, v in G.edges():
    if u in node2idx and v in node2idx:
        edges.append((node2idx[u], node2idx[v]))

# 转为 PyG 的 edge_index
edge_index = torch.from_numpy(np.array(edges, dtype=np.int64)).t().contiguous()

# 构建图数据
data = Data(x=x, edge_index=edge_index)


model.eval()
with torch.no_grad():
    z = model(data.x, data.edge_index)

# 转换为 NumPy 数组
z_np = z.cpu().numpy()

z_np = PCA(n_components=100).fit_transform(z_np)

# 使用 HDBSCAN 进行聚类
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=15,
    min_samples=5,
    cluster_selection_method="eom",
    cluster_selection_epsilon=0.2,
    metric="euclidean",
    gen_min_span_tree=True,
)

labels = clusterer.fit_predict(z_np)  # 聚类标签

# 统计聚类情况
n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)  # 排除噪声点（-1）
print(f"Detected {n_clusters} clusters with HDBSCAN.")
print(f"Noise points: {np.sum(labels == -1)}")

# 使用 t-SNE 降维到 2D 以便可视化
z_2d = TSNE(n_components=2, perplexity=50, random_state=42).fit_transform(z_np)

# 绘制聚类结果
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    z_2d[:, 0], 
    z_2d[:, 1], 
    c=labels, 
    cmap='Spectral',  # 适合多类别的颜色映射
    s=10, 
    alpha=0.8
)
plt.colorbar(scatter, label='Cluster ID')
plt.title(f"HDBSCAN Clustering (t-SNE projection)\n{n_clusters} clusters, {np.sum(labels == -1)} noise points")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.savefig('./image/tsne_GNN_HDBSCAN.png', dpi=300, bbox_inches='tight')