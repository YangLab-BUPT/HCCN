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
from gseapy import enrichr
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel


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

ontology_name = 'mfo'


# # 给 valid_nodes 中每个节点打上 MFO/BPO/CCO 标签（如果缺失就跳过）
# z_array = z.cpu().numpy()
# z_2d = TSNE(n_components=2, perplexity=50).fit_transform(z_array)

# ns_labels = []
# z_2d_filtered = []

# for i, go_id in enumerate(valid_nodes):
#     ns = go_id2ns.get(go_id)
#     if ns in ns_label_map:
#         ns_labels.append(ns_label_map[ns])
#         z_2d_filtered.append(z_2d[i])

# z_2d_filtered = np.array(z_2d_filtered)
# ns_labels = np.array(ns_labels)


ontology_mask = np.array([go_id2ns.get(go_id) == 'molecular_function' for go_id in valid_nodes])
ontology_z = z[ontology_mask].cpu().numpy()
ontology_go_ids = [go_id for i, go_id in enumerate(valid_nodes) if ontology_mask[i]]


# X = ontology_z

# sse = []
# cluster_range = list(range(2, 61))  # 尝试聚类数从 2 到 60

# for k in cluster_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
#     kmeans.fit(X)
#     sse.append(kmeans.inertia_)  # inertia_ 就是 SSE

# # 画图
# plt.figure(figsize=(8, 5))
# plt.plot(cluster_range, sse, 'o-', linewidth=2)
# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("Sum of Squared Errors (SSE)")
# plt.title("Elbow Method for Optimal k (KMeans)")
# plt.grid(True)
# plt.savefig(f'./image/elbow_{ontology_name}_kmeans.png', dpi=300, bbox_inches='tight')


n_clusters = 8

# # 进行谱聚类
# similarity_matrix = rbf_kernel(mfo_z, gamma=0.1)
# spectral = SpectralClustering(n_clusters=n_clusters, 
#                               affinity='precomputed', 
#                               assign_labels='kmeans', 
#                               random_state=42)

# cluster_labels = spectral.fit_predict(similarity_matrix)

# 使用 Kmeans 聚类
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(ontology_z)


ontology_z_2d = TSNE(n_components=2, perplexity=30).fit_transform(ontology_z)

plt.figure(figsize=(12, 10))
plt.scatter(ontology_z_2d[:, 0], ontology_z_2d[:, 1], 
            c=cluster_labels, cmap='tab20', 
            s=20, alpha=0.8)
plt.colorbar(label='Cluster ID')
plt.title(f"{ontology_name.upper()} KMeans Clustering (t-SNE)\n{n_clusters} clusters, {len(ontology_z)} terms")
plt.savefig(f'./image/tsne_{ontology_name}_kmeans_clusters.png', dpi=300, bbox_inches='tight')




# TODO 查看富集程度

import numpy as np
from collections import defaultdict, Counter
from pronto import Ontology
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


def get_direct_parents(term):
    """返回 term 的直接 is_a 父项（基于 superclasses 差分）"""
    all_supers = set(term.superclasses(with_self=False))
    direct_parents = set()
    for candidate in all_supers:
        if not any(
            candidate != other and candidate in other.superclasses(with_self=False)
            for other in all_supers
        ):
            direct_parents.add(candidate)
    return direct_parents


def get_ancestors(go, go_id, level_limit=3):
    """获取给定 GO term 的 is_a 上位类（限制层数）"""
    if go_id not in go:
        return set()

    term = go[go_id]
    ancestors = set()
    visited = set()

    def recurse(t, level):
        if level >= level_limit:
            return
        for parent in get_direct_parents(t):
            if parent.id not in visited:
                visited.add(parent.id)
                ancestors.add(parent.id)
                recurse(parent, level + 1)

    recurse(term, 0)
    return ancestors


def cluster_go_enrichment(go, go_ids, cluster_labels, level_limit=3, topk=10):
    """
    输入：
        go: pronto.Ontology 对象
        go_ids: List[str]，每个 term 的 GO ID，顺序与 cluster_labels 一一对应
        cluster_labels: List[int]，每个 term 所属的聚类编号
        level_limit: 向上取多少层 ancestor（默认 3）
        topk: 返回每个 cluster top 富集的 ancestor 数量

    返回：
        dict[cluster_id] = List[Tuple[ancestor_id, name, pval, count_in_cluster, count_in_total]]
    """
    all_ancestors = {}
    for go_id in go_ids:
        all_ancestors[go_id] = get_ancestors(go, go_id, level_limit=level_limit)

    # 聚类划分
    cluster_terms = defaultdict(list)
    for go_id, label in zip(go_ids, cluster_labels):
        cluster_terms[label].append(go_id)

    # 全体统计
    bg_ancestor_counter = Counter()
    for go_id in go_ids:
        bg_ancestor_counter.update(all_ancestors[go_id])
    total_terms = len(go_ids)

    enriched = {}

    for label, terms in cluster_terms.items():
        fg_ancestor_counter = Counter()
        for go_id in terms:
            fg_ancestor_counter.update(all_ancestors[go_id])
        fg_total = len(terms)

        results = []
        for ancestor, fg_count in fg_ancestor_counter.items():
            bg_count = bg_ancestor_counter[ancestor]
            if fg_count == 0 or bg_count == 0:
                continue
            a = fg_count
            b = fg_total - a
            c = bg_count - a
            d = total_terms - fg_total - c
            table = [[a, b], [c, d]]
            _, pval = fisher_exact(table, alternative='greater')
            results.append((ancestor, pval, a, bg_count))

        # 多重检验校正
        if results:
            pvals = [r[1] for r in results]
            _, padjs, _, _ = multipletests(pvals, method='fdr_bh')
            results = [
                (ancestor, go[ancestor].name if ancestor in go else "?", pval, a, bg_count, padj)
                for (ancestor, pval, a, bg_count), padj in zip(results, padjs)
            ]
            results.sort(key=lambda x: x[-1])  # 按 padj 排序
            enriched[label] = results[:topk]
        else:
            enriched[label] = []

    return enriched


enriched = cluster_go_enrichment(go, ontology_go_ids, cluster_labels, level_limit=3, topk=10)

# 打印每个聚类的 top 富集 GO term
for cluster_id, results in enriched.items():
    print(f"\nCluster {cluster_id}:")
    for go_id, name, pval, in_cluster, in_total, padj in results:
        print(f"  {go_id} ({name}): p={pval:.2e}, padj={padj:.2e}, {in_cluster}/{in_total}")


# 保存聚类结果
import pandas as pd
cluster_df = pd.DataFrame({
    'GO_ID': ontology_go_ids,
    'Cluster_ID': cluster_labels
})

cluster_df.to_csv(f'{ontology_name}_kmeans_clusters.csv', index=False)
print(f"聚类结果已保存为 {ontology_name}_kmeans_clusters.csv")