import pandas as pd
from pymilvus import MilvusClient,FieldSchema, CollectionSchema, DataType, Collection
from tqdm import tqdm
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer, BertTokenizer, BertForMaskedLM
import itertools
from os.path import join


client = MilvusClient("./db/milvus.db")
if client.has_collection(collection_name="train"):   # 如果存在则删除
    client.drop_collection(collection_name="train")
    client.drop_collection(collection_name="test")

path = '/your/root/path'
datatype = 'stride'
model_path = '/data/gxyData/prot_bert'   # Your embedding model path
train_data_dir1 = join(path, datatype, 'traindata.csv')
train_data_dir2 = join(path, datatype, 'trainNolabel.csv')
# train_data_dir3 = path + 'traindata2.csv'
train_filelist = [train_data_dir1, train_data_dir2]
train_layer_data = join(path, datatype, 'trainlayerdata.csv')

test_data_dir1 = join(path, datatype, 'testdata.csv')
test_data_dir2 = join(path, datatype, 'testNolabel.csv')

test_filelist = [test_data_dir1, test_data_dir2]
test_layer_data = join(path, datatype, 'trainlayerdata.csv')

tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
model = BertForMaskedLM.from_pretrained(model_path).to('cuda')

for _, param in model.named_parameters():
    param.requires_grad = False

schema = CollectionSchema(fields=[FieldSchema(name="index", dtype=DataType.INT64, is_primary=True, auto_id=False), 
                                  FieldSchema(name="GOcluster", dtype=DataType.FLOAT_VECTOR, dim=630),   # change to the cluster num
                                  FieldSchema(name="GO", dtype=DataType.FLOAT_VECTOR, dim=7962),   # change to the node num
                                  FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4096)], # 512 * 2 * 4
                        description="Example collection")

client.create_collection(
    collection_name="train",
    schema=schema
)

client.create_collection(
    collection_name="test",
    schema=schema
)

batch_size = 64

def extract_features(layer_output, index):
    mean_feat = layer_output.mean(dim=-1)[index].to('cpu').numpy().tolist()
    max_feat = layer_output.max(dim=-1).values[index].to('cpu').numpy().tolist()
    return mean_feat, max_feat

def write_data2milvus(data, cluster_data, TABLE_NAME):

    for start_idx in tqdm(range(0, data.shape[0], batch_size)):
        end_idx = min(start_idx + batch_size, data.shape[0])
        batch_data = data.iloc[start_idx:end_idx]
        batch_cluster_data = cluster_data.iloc[start_idx:end_idx]
        texts = batch_data.iloc[:, 1].tolist()  # 获取文本列
        inputs = tokenizer(texts, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
        layers = [6, 12, 20, -1]
        hidden_states = model(**inputs, output_hidden_states=True).hidden_states

        insert_data = []
        for index in range(end_idx - start_idx):
            embedding = list(itertools.chain.from_iterable(
                extract_features(hidden_states[layer], index) for layer in layers
            ))
            temp_data = {
                'index': index + start_idx,
                'GOcluster': batch_cluster_data.iloc[index, 1:].astype(np.float16).tolist(),
                'GO': batch_data.iloc[index, 2:].astype(np.float16).tolist(),
                'embedding': embedding
            }
            insert_data.append(temp_data)
        client.insert(collection_name=TABLE_NAME, data=insert_data)

traindata_list = []
for i in train_filelist:
    traindata_list.append(pd.read_csv(i))

data = pd.concat(traindata_list).reset_index(drop=True)
cluster_data = pd.read_csv(path + datatype + 'trainlayerdata.csv')
write_data2milvus(data, cluster_data, "train")

validdata_list = []
for i in test_filelist:
    validdata_list.append(pd.read_csv(i))

data = pd.concat(validdata_list).reset_index(drop=True)
cluster_data = pd.read_csv(path + datatype + 'testlayerdata.csv')
write_data2milvus(data, cluster_data, "test")