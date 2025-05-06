import json
import pandas as pd
 

# 三元组：
with open('/data/csydata/ragflow/outputs/deepseek-chat_models_GritLM_GritLM-7B/openie_results_ner_deepseek-chat.json') as f:
    data = json.load(f)
 
print('三元组：',data)

# 读取 Parquet 文件
pqlist = {'chunk_embeddings':'vdb_chunk','entity_embeddings':'vdb_entity','fact_embeddings':'vdb_fact'}
for pq,vdb in pqlist.items():
    data = pd.read_parquet(f'/data/csydata/ragflow/outputs/deepseek-chat_models_GritLM_GritLM-7B/{pq}/{vdb}.parquet')
    print(data)  # emb, 文档的向量是4096维度

# 读取 pickle 文件
pkdata = pd.read_pickle('/data/csydata/ragflow/outputs/deepseek-chat_models_GritLM_GritLM-7B/graph.pickle')
print(pkdata)