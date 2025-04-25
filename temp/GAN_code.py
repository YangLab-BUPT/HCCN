import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 加载数据
node2vec_data = pd.read_csv('godata/go_node2vec_embeddings.csv')
node2vec_ids = list(node2vec_data['GO_ID'])
del node2vec_data['GO_ID']
t_node2vec_emb = node2vec_data.values
bert_emb = np.load('godata/go_sapbert_cls_embeddings.npy')
bert_ids = list(pd.read_csv('godata/go_sapbert_ids.csv')['GO_ID'])

id_to_idx_node2vec = {go_id: idx for idx, go_id in enumerate(node2vec_ids)}

node2vec_emb = np.array([t_node2vec_emb[id_to_idx_node2vec[go_id]] for go_id in bert_ids])

# 校验数据一致性
assert len(node2vec_emb) == len(bert_emb), "样本数量必须一致！"
print(f"数据维度：Node2Vec={node2vec_emb.shape}, BERT={bert_emb.shape}")

# 预处理
scaler_node2vec = StandardScaler()
scaler_bert = StandardScaler()

node2vec_norm = scaler_node2vec.fit_transform(node2vec_emb)
bert_norm = scaler_bert.fit_transform(bert_emb)

node2vec_tensor = torch.FloatTensor(node2vec_norm)
bert_tensor = torch.FloatTensor(bert_norm)

# 模型定义
class ProjectionNet(nn.Module):
    """动态适配输入/输出维度"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 768),
            nn.LeakyReLU(0.2),
            nn.Linear(768, output_dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    """自动适配BERT的嵌入维度"""
    def __init__(self, output_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x)

# 根据数据维度初始化模型
input_dim = node2vec_emb.shape[1]
output_dim = bert_emb.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = ProjectionNet(input_dim, output_dim).to(device)
D = Discriminator(output_dim).to(device)


# 定义优化器和损失函数
opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-5, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# 创建数据加载器
dataset = torch.utils.data.TensorDataset(node2vec_tensor, bert_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

num_epochs = 400
for epoch in range(num_epochs):
    for i, (node2vec_batch, bert_batch) in enumerate(dataloader):
        real_emb = bert_batch.to(device)
        node2vec_batch = node2vec_batch.to(device)
        
        # 训练判别器
        D.zero_grad()
        
        # 真实样本的损失
        real_labels = torch.ones(real_emb.size(0), 1).to(device)
        real_loss = criterion(D(real_emb), real_labels)
        
        # 生成样本的损失
        fake_emb = G(node2vec_batch).detach()
        fake_labels = torch.zeros(fake_emb.size(0), 1).to(device)
        fake_loss = criterion(D(fake_emb), fake_labels)
        
        # 判别器损失
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        opt_D.step()
        
        # 训练生成器
        G.zero_grad()
        
        # 生成器损失
        fake_emb = G(node2vec_batch)
        g_loss = criterion(D(fake_emb), real_labels)
        
        cosine_sim = torch.cosine_similarity(fake_emb, real_emb, dim=1).mean() * 0.5
        g_loss += 2 * (1 - cosine_sim)
        
        g_loss.backward()
        opt_G.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{num_epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | CosSim: {cosine_sim.item():.4f}")

# 保存对齐后的嵌入
with torch.no_grad():
    aligned_node2vec = G(node2vec_tensor.to(device)).cpu().numpy()
alpha = 0.6 
fused_emb = alpha * aligned_node2vec + (1 - alpha) * bert_norm

np.save('godata/go_aligned_embeddings.npy', fused_emb)

# 保存判别器和生成器
torch.save(D.state_dict(), 'D.pth')
torch.save(G.state_dict(), 'G.pth')

