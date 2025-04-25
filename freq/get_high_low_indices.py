import json
import pandas as pd
from os.path import join

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

temp_go = list(pd.read_csv(test_data_dir2, nrows=0).columns)[2:]

# 得到 根 -> 叶子 字典
with open('/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/bpo/GO_bpo_withlayerDepthDict3.json') as f:
    temp_hierar_relations = json.load(f)
    for go in temp_hierar_relations:
        temp_hierar_relations[go] = temp_hierar_relations[go]['leaf']

# 得到 分层聚类字典
with open('/data/csyData/pygosemsim-master/pygosemsim-master/bpo_Kmeans_cluster2.json') as f:
    temp_cluster_nodes = json.load(f)

# 得到alpha
with open('/data/csyData/uniprot_test/code/GOcode/bpo_version2/freq.json') as f:
    freq = json.load(f)

def go2index(target_go):
    go = temp_go
    if isinstance(target_go, list):
        return [go.index(i) for i in target_go]
    elif isinstance(target_go, str):
        if target_go in go:
            return go.index(target_go)
        else:
            return None


node_nums = len(temp_hierar_relations.keys()) - 1
layer_nums = len(temp_cluster_nodes.keys())
labels_num = node_nums + layer_nums

high_freq = {}
low_freq = {}
hierar_relations = {}

alpha = [0] * node_nums

for go,value in freq.items():
    if go in temp_go:
        alpha[go2index(go)] = value
    if value < 0.7:
        high_freq[go] = value
    elif value > 0.8:
        low_freq[go] = value

def high_low_other_indices(data, output):
    high_freq_label_indices = data[data[list(high_freq.keys())].eq(1).any(axis=1) & data[list(low_freq.keys())].eq(0).all(axis=1)].index
    low_freq_label_indices = data[data[list(low_freq.keys())].eq(1).any(axis=1)].index.difference(high_freq_label_indices)
    
    other_indices = data.index.difference(low_freq_label_indices).difference(high_freq_label_indices)
    with open(output+"_high_freq_label_indices.json", "w+") as jsonFile:
        jsonFile.write(json.dumps(list(high_freq_label_indices)))
    with open(output+"_low_freq_label_indices.json", "w+") as jsonFile:
        jsonFile.write(json.dumps(list(low_freq_label_indices)))
    with open(output+"_other_indices.json", "w+") as jsonFile:
        jsonFile.write(json.dumps(list(other_indices)))

traindata_list = []
for i in train_filelist:
    traindata_list.append(pd.read_csv(i))
data = pd.concat(traindata_list,axis=0).reset_index(drop=True)

high_low_other_indices(data, 'train')

validdata_list = []
for i in test_filelist:
    validdata_list.append(pd.read_csv(i))
data = pd.concat(validdata_list,axis=0).reset_index(drop=True)

high_low_other_indices(data, 'valid')