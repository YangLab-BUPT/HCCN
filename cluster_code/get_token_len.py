from transformers import AutoTokenizer
from pronto import Ontology
import matplotlib.pyplot as plt

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("/data/csydata/ragflow/models/cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

# 加载 GO 本体
go = Ontology("./godata/go-basic-2311.obo")

with_path_lengths = []
no_path_lengths = []

# 遍历 GO term
for term in go.terms():
    if term.obsolete:
        continue

    text_no_path = f"{term.name}: {term.definition or ''}"
    no_path_len = len(tokenizer.tokenize(text_no_path))
    no_path_lengths.append(no_path_len)

    # 加入 path 的文本（只取前3层）
    path = ' > '.join([t.name for t in list(term.superclasses(with_self=False))[:3]])
    text_with_path = f"{term.name}: {term.definition or ''} [PATH] {path}"
    with_path_len = len(tokenizer.tokenize(text_with_path))
    with_path_lengths.append(with_path_len)

# 画出分布对比
plt.figure(figsize=(10, 6))
plt.hist(no_path_lengths, bins=50, alpha=0.6, label="Without PATH")
plt.hist(with_path_lengths, bins=50, alpha=0.6, label="With PATH (top 3 levels)")
plt.axvline(x=64, color='red', linestyle='--', label="max_length=64")
plt.title("Token Length Distribution of GO Term Texts")
plt.xlabel("Number of Tokens")
plt.ylabel("Number of Terms")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./image/token_len.png')
