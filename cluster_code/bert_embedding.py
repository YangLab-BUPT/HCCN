from transformers import AutoTokenizer, AutoModel
from pronto import Ontology
import numpy as np
import torch
from tqdm import tqdm

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("/data/csydata/ragflow/models/cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
model = AutoModel.from_pretrained("/data/csydata/ragflow/models/cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda()
model.eval()

# 准备文本列表
go = Ontology("./godata/go-basic-2311.obo")
all_names = []
go_ids = []

for term in go.terms():
    if term.obsolete:
        continue
    path = ' > '.join([t.name for t in list(term.superclasses(with_self=True))[:3]])
    text = f"{term.name}: {term.definition or ''} [PATH] {path}"
    all_names.append(text)
    go_ids.append(term.id)

# 批量生成嵌入
bs = 64  # 若显存足够可调大
all_embs = []

for i in tqdm(range(0, len(all_names), bs)):
    toks = tokenizer.batch_encode_plus(
        all_names[i:i+bs], padding="max_length", max_length=300,
        truncation=True, return_tensors="pt"
    )
    toks = {k: v.cuda() for k, v in toks.items()}
    with torch.no_grad():
        cls_rep = model(**toks)[0][:, 0, :]  # 使用 [CLS] token
    all_embs.append(cls_rep.cpu().numpy())

# 合并并保存
all_embs = np.concatenate(all_embs, axis=0)
np.save("godata/go_sapbert_cls_embeddings.npy", all_embs)

import csv
with open("godata/go_sapbert_ids.csv", "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["GO_ID"])
    writer.writerows([[go_id] for go_id in go_ids])
