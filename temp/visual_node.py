import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 加载 BERT 向量和 ID
bert_emb = np.load('godata/go_sapbert_cls_embeddings.npy')  # (N, 768)
bert_ids = list(pd.read_csv('godata/go_sapbert_ids.csv')['GO_ID'])  # 长度 N

# 加载 Node2Vec的向量
node2vec_data = pd.read_csv('godata/go_node2vec_embeddings.csv')  # 假设CSV无表头，直接读取数值
node2vec_ids = list(node2vec_data['GO_ID'])
del node2vec_data['GO_ID']
t_node2vec_emb = node2vec_data.values
id_to_idx_node2vec = {go_id: idx for idx, go_id in enumerate(node2vec_ids)}
node2vec_emb = np.array([t_node2vec_emb[id_to_idx_node2vec[go_id]] for go_id in bert_ids])

# 构建 GO ID -> embedding 的映射
id2emb = {go_id: emb for go_id, emb in zip(bert_ids, bert_emb)}


# 构建 ID 到 Node2Vec 向量的映射
id2vec = {go_id: emb for go_id, emb in zip(node2vec_ids, node2vec_emb)}

# 只拼接那些同时出现在 BERT 和 Node2Vec 中的 ID
shared_ids = [go_id for go_id in bert_ids if go_id in id2vec]

# 拼接向量
# concat_emb = np.array([
#     np.concatenate([id2emb[go_id], id2vec[go_id]])
#     for go_id in shared_ids
# ])  # shape: (M, 896)

# 聚类（可选，用于配色）
labels_concat = KMeans(n_clusters=10).fit_predict(node2vec_emb)

# t-SNE 降维
z_concat_2d = TSNE(n_components=2, perplexity=50).fit_transform(node2vec_emb)

# 可视化
plt.figure(figsize=(10, 8))
plt.scatter(z_concat_2d[:, 0], z_concat_2d[:, 1], c=labels_concat, cmap='tab10', s=10)
plt.title("concat BERT + Node2Vec t-SNE visualization")
plt.savefig('./image/node2vec_tsne.png')