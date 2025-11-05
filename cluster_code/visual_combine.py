import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from GNN_net import GraphSAGE
import pickle
import random
from matplotlib.gridspec import GridSpec
from pronto import Ontology
from os.path import join, exists

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

device = torch.device('cuda:1')

root = '/data/csyData/uniprot_test/code/GOcode/upload/HCCN/HCCN/cluster_code'
# === 加载数据 ===
go = Ontology(f"{root}/godata/go-basic-2311.obo")

model = GraphSAGE(768, 256, 128).to(device)
model.load_state_dict(torch.load(f"{root}/GNN_net.pth"))

bert_emb = np.load(f"{root}/godata/go_sapbert_cls_embeddings.npy")  # (N, 768)
bert_ids = list(pd.read_csv(f"{root}/godata/go_sapbert_ids.csv")['GO_ID'])
id2emb = {go_id: emb for go_id, emb in zip(bert_ids, bert_emb)}

with open(f"{root}/godata/go_subgraph.pkl", "rb") as f:
    G = pickle.load(f)

node_list = list(G.nodes())
valid_nodes = [n for n in node_list if n in id2emb]

go_id2ns = {
    go_id: go[go_id].namespace
    for go_id in valid_nodes
    if go_id in go
}

x = torch.tensor([id2emb[n] for n in valid_nodes], dtype=torch.float)
node2idx = {node: idx for idx, node in enumerate(valid_nodes)}
edges = [(node2idx[u], node2idx[v]) for u, v in G.edges() if u in node2idx and v in node2idx]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
data = Data(x=x, edge_index=edge_index).to(device)

# === 前向推理 ===
model.eval()
with torch.no_grad():
    z = model(data.x, data.edge_index)
z_array = z.cpu().numpy()

# === namespace 标签 ===
ns_label_map = {
    'molecular_function': 0,
    'biological_process': 1,
    'cellular_component': 2
}
ns_names = ['MFO', 'BPO', 'CCO']
colors = ['#1f77b4', '#2ca02c', '#d62728']

# === t-SNE (统一嵌入) ===
z_2d = TSNE(n_components=2, perplexity=50, random_state=42).fit_transform(z_array)

# === 各 ontology 数据 ===
plots = {}
for ont, ns_name in zip(['molecular_function', 'biological_process', 'cellular_component'], ns_names):
    mask = np.array([go_id2ns.get(gid) == ont for gid in valid_nodes])
    sub_z = z_array[mask]
    sub_z2d = z_2d[mask]
    if len(sub_z) > 100:
        n_clusters = 8
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(sub_z)
        plots[ns_name] = (sub_z2d, cluster_labels)
    else:
        print(f"⚠️ {ns_name} too few nodes: {len(sub_z)}")

# === Namespace 总体分布 ===
ns_labels, ns_points = [], []
for i, go_id in enumerate(valid_nodes):
    ns = go_id2ns.get(go_id)
    if ns in ns_label_map:
        ns_labels.append(ns_label_map[ns])
        ns_points.append(z_2d[i])
plots['Namespace'] = (np.array(ns_points), np.array(ns_labels))

# === 绘图风格 ===
# plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.0  # 边框线宽
plt.rcParams['axes.edgecolor'] = 'black'

fig, axes = plt.subplots(2, 2, figsize=(16, 15), dpi=400)
axes = axes.flatten()

labels = ['A', 'B', 'C', 'D']
plot_names = ['MFO', 'BPO', 'CCO', 'Namespace']

for i, name in enumerate(plot_names):
    ax = axes[i]
    X, y = plots[name]  # 这里 plots[name] 是对应的 t-SNE 数据
    
    if name == 'Namespace':
        for j, nsn in enumerate(ns_names):
            idx = y == j
            ax.scatter(X[idx, 0], X[idx, 1], c=colors[j], label=nsn, s=8)
        ax.legend(markerscale=2, fontsize=9, loc='best', frameon=False)
    else:
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab20', s=8, alpha=0.8)
    
    # 添加外框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)
    
    # 添加标题和标签
    label = labels[i]
    ax.set_title(f"({label}) {name} KMeans Clustering (t-SNE)", fontsize=15, pad=6)
    # ax.text(-0.1, 1.08, labels[i], transform=ax.transAxes,
    #         fontsize=16, fontweight='bold', va='top', ha='right')
    
    # 去除刻度但保留外框
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig(f"{root}/image/GNN_tsne_all_combined_labeled.png", dpi=400, bbox_inches='tight')
plt.show()
