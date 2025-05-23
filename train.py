import json
import time
import torch
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from ProtTransModel3 import ClassConfig, LayerLoss, NodeLoss, T5EncoderCLSModel2, Myconfig
from accelerate.logging import get_logger
from sklearn.metrics import roc_curve, auc
from safetensors.torch import load_file
from MySampler import CommonLabelSampler, CycleLabelSampler, SelectBatchSampler
from accelerate import Accelerator, DeepSpeedPlugin
from pymilvus import MilvusClient

logger = get_logger(__name__)
# os.environ["ACCELERATE_TORCH_DEVICE"] = "cuda:1"
torch.autograd.set_detect_anomaly(True)

model_dirname = '/step_Focalloss6_'
model_path = '/data/csyData/prot_t5_xl_half_uniref50-enc'

with open('/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/bpo/label_freq_list.json') as f:
    label_freq_list = json.load(f)

path = '/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/bpo/withlayer/'
datatype = 'stride/'
train_data_dir1 = path + datatype + 'traindata.csv'
train_data_dir2 = path + datatype + 'trainNolabel.csv'
# train_data_dir3 = path + 'traindata2.csv'
train_filelist = [train_data_dir1, train_data_dir2]
train_layer_data = path + datatype + 'trainlayerdata.csv'

test_data_dir1 = path + datatype + 'testdata.csv'
test_data_dir2 = path + datatype + 'testNolabel.csv'
# test_data_dir3 = path + 'testdata2.csv'
test_filelist = [test_data_dir1, test_data_dir2]
test_layer_data = path + datatype + 'testlayerdata.csv'


# 连接数据库
client = MilvusClient("./db/milvus.db")


# 得到 根 -> 叶子 字典
with open('/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/bpo/GO_bpo_withlayerDepthDict3.json') as f:
    temp_hierar_relations = json.load(f)
    for go in temp_hierar_relations:
        temp_hierar_relations[go] = temp_hierar_relations[go]['leaf']

# 得到 分层聚类字典
with open('/data/csyData/pygosemsim-master/pygosemsim-master/bpo_Kmeans_cluster2.json') as f:
    temp_cluster_nodes = json.load(f)

# 得到 聚类的父子关系字典
# with open('/data/csyData/uniprot_test/code/GOcode/cco_version2/cco_cluster_relations.json') as f:
with open('/data/csyData/pygosemsim-master/pygosemsim-master/bpo_KMeanscluster_relations.json') as f:
    cluster_relations = json.load(f)

# 得到标签频率按照阈值分成四份的字典
# with open('/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/cco/label_freq_list.json') as f:
#     label_freq_list = json.load(f)

# 得到标签频率按照字典
# with open('/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/cco/sta_count3.json') as f:
#     sta_count3 = json.load(f)

# 得到alpha
with open('/data/csyData/uniprot_test/code/GOcode/bpo_version2/freq.json') as f:
    node_freq = json.load(f)

with open('/data/csyData/uniprot_test/code/GOcode/bpo_version2/freq2.json') as f:
    layer_freq = json.load(f)

node_nums = len(temp_hierar_relations.keys()) - 1
layer_nums = len(temp_cluster_nodes.keys())
labels_num = node_nums + layer_nums

epoch = 800    # 可能修改
max_length = 512
train_batch_size = valid_batch_size = 512
# lr = 3e-4
# lr = 1e-3
# only_layer_metrics = False
load_model = False
had_get_layer = True
# saved_ep = 50   # 重新训练时，记得改成最新的模型参数
saved_ep = 0

temp_go = list(pd.read_csv(test_data_dir2, nrows=0).columns)[2:]
temp_layer = list(pd.read_csv(path + datatype + 'trainlayerdata.csv', nrows=0).columns)[1:]
def layer2index(layer):
    layername = temp_layer
    if isinstance(layer, list):
        return [layername.index(i) for i in layer]
    elif isinstance(layer, str):
        if layer in layername:
            return layername.index(layer)
        else:
            return None

def go2index(target_go):
    go = temp_go
    if isinstance(target_go, list):
        return [go.index(i) for i in target_go]
    elif isinstance(target_go, str):
        if target_go in go:
            return go.index(target_go)
        else:
            return None

def min_max_normalize(lst):
    min_val = min(lst)
    max_val = max(lst)
    return [round((x - min_val) / (max_val - min_val), 3) for x in lst]

# cluster_nodes = [temp_cluster_nodes[key] for key in sorted(temp_cluster_nodes)]  
main_numbers, sub_numbers = [], []
for label in list(temp_cluster_nodes.keys()):
    if '_' not in label:
        main_numbers.append(label)
    else:
        sub_numbers.append(label)

main_numbers.sort(key=lambda x: int(x))
sub_numbers.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])) if '_' in x else (int(x), -1))
cluster_nodes = main_numbers + sub_numbers


# high_freq = {}
# low_freq = {}
hierar_relations = {}

for node,value in temp_hierar_relations.items():
    if go2index(node):
        hierar_relations[go2index(node)] = go2index(value)

cluster_nodes_relations = {}
for index,value in enumerate(cluster_nodes):
    cluster_nodes_relations[str(index)] = go2index(temp_cluster_nodes[value])

node_alpha = [0] * node_nums
layer_alpha = [0] * layer_nums

for i,value in layer_freq.items():
    if i in temp_layer:
        layer_alpha[layer2index(i)] = value
    else:
        print(i)

for i,value in node_freq.items():
    if i in temp_go:
        node_alpha[go2index(i)] = value


class MyDataset(Dataset):

    def __init__(self, indices_name) -> None:
        super().__init__()
        # self.data = pd.concat(data_dirlist,axis=0).reset_index(drop=True)
        # if had_get_layer and layer_dirlist is not None:
        #     self.layer_data = pd.read_csv(layer_dirlist).reset_index(drop=True)
        self.client = client
        self.collection_name = indices_name

        # self.go = list(self.data.columns)[2:]
        with open(indices_name + '_high_freq_label_indices.json') as f:
            self.high_freq_label_indices = json.load(f)
        with open(indices_name + '_low_freq_label_indices.json') as f:
            self.low_freq_label_indices = json.load(f)
        with open(indices_name + '_other_indices.json') as f:
            self.other_indices = json.load(f)

        # self.high_freq_label_indices = self.data[self.data[list(high_freq.keys())].eq(1).any(axis=1) & self.data[list(low_freq.keys())].eq(0).all(axis=1)].index
        # self.low_freq_label_indices = self.data[self.data[list(low_freq.keys())].eq(1).any(axis=1)].index.difference(self.high_freq_label_indices)
        # self.other_indices = self.data.index.difference(self.low_freq_label_indices).difference(self.high_freq_label_indices)

    def __getitem__(self, index):
        expr = f"index == {index}"
        search_res = self.client.query(
            collection_name = self.collection_name,
            filter=expr,
            output_fields=["GOcluster", "GO", "embedding"],
            limit=1
        )[0]

        return search_res['embedding'], search_res['GOcluster'], search_res['GO']

    def __len__(self):
        return self.client.get_collection_stats(self.collection_name)['row_count']
    
    # def go2index(self,target_go):
    #     go = self.go
    #     if isinstance(target_go, list):
    #         return [go.index(i) for i in target_go]
    #     elif isinstance(target_go, str):
    #         if target_go in go:
    #             return go.index(target_go)
    #         else:
    #             return None

class MyDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(MyDataLoader, self).__init__(*args, **kwargs)
        self.go = go

    def go2index(self,target_go):
        if isinstance(target_go, list):
            return [self.go.index(i) for i in target_go]
        elif isinstance(target_go, str):
            if target_go in self.go:
                return self.go.index(target_go)
            else:
                return None
        
def prepare_dataloader(accelerator):

    trainset = MyDataset('train')

    validset = MyDataset('valid')

    # tokenizer = T5Tokenizer.from_pretrained(model_path)

    def collate_func(batch):
        emb, layers, nodes = [], [], []
        inputs = {}
        for item in batch:
            emb.append(item[0])
            layers.append(item[1])
            nodes.append(item[2])
        # inputs = tokenizer(emb, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        # del inputs["token_type_ids"]
        inputs["emb"] = torch.tensor(np.stack(emb)).to('cuda').to(torch.bfloat16)
        inputs["nodes"] = torch.tensor(np.stack(nodes)).to('cuda').to(torch.bfloat16)
        inputs["layers"] = torch.tensor(np.stack(layers)).to('cuda').to(torch.bfloat16)
        return inputs

    high_freq_sampler = CommonLabelSampler(trainset.high_freq_label_indices, \
                                    num_replicas=accelerator.num_processes, \
                                    rank=accelerator.process_index, \
                                    shuffle=True, \
                                    drop_last=True)

    low_freq_sampler = CycleLabelSampler(trainset.low_freq_label_indices, \
                                        num_replicas=accelerator.num_processes, \
                                        rank=accelerator.process_index, \
                                        shuffle=True, \
                                        drop_last=False)

    other_sampler = CycleLabelSampler(trainset.other_indices, \
                                        num_replicas=accelerator.num_processes, \
                                        rank=accelerator.process_index, \
                                        shuffle=True, \
                                        drop_last=False)

    train_batch_sampler = SelectBatchSampler(high_freq_sampler, 
                                            low_freq_sampler,
                                            other_sampler,
                                            batch_size=valid_batch_size,
                                            drop_last=True)


    trainloader = MyDataLoader(trainset, 
                            collate_fn=collate_func, 
                            batch_sampler=train_batch_sampler)

    # trainloader = MyDataLoader(trainset, batch_size=train_batch_size, collate_fn=collate_func, shuffle=True)
    validloader = MyDataLoader(validset, batch_size=valid_batch_size, collate_fn=collate_func, shuffle=False)

    return trainloader, validloader

# prepare
def prepare_model_and_optimizer(accelerator):

    # if os.path.exists('high_freq.json'):
    #     with open('high_freq.json') as f:
    #         high_freq = json.load(f)
    #     with open('low_freq.json') as f:
    #         low_freq = json.load(f)
    # else:
    #     for go,value in freq.items():
    #         alpha[validloader.go2index(go)] = value
    #         if value < 0.4:
    #             high_freq[go] = value
    #         elif value > 0.94:
    #             low_freq[go] = value
    #     with open("high_freq.json", "w+") as jsonFile:
    #         jsonFile.write(json.dumps(high_freq, indent = 4))
    #     with open("low_freq.json", "w+") as jsonFile:
    #         jsonFile.write(json.dumps(low_freq, indent = 4))


    # model = T5EncoderModel.from_pretrained(model_path)
    # model.config.d_model = 1024

    # lora_target_modules = []
    # for name, _ in model.named_modules():
    #     for layer in [23]:
    #         if (str(layer) in name) & (('.wi' in name) or ('.wo' in name) or ('.q' in name) or ('.k' in name) or ('.v' in name) or ('.o' in name)):
    #             lora_target_modules.append(name)

    # peft_config = LoraConfig(
    #                 lora_alpha=16,
    #                 lora_dropout=0.1,
    #                 r=8,
    #                 bias="none",
    #                 task_type="SEQ_2_SEQ_LM",
    #                 target_modules=lora_target_modules
    #                 )
    # model = prepare_model_for_kbit_training(model)
    # model = get_peft_model(model, peft_config)

    config = Myconfig()

    class_config = ClassConfig(node_nums=node_nums, layer_nums=layer_nums, batch_size=train_batch_size)

    model = T5EncoderCLSModel2(config,
                                class_config,
                                cluster_relations,
                                hierar_relations,
                                cluster_nodes_relations,
                                main_numbers,
                                sub_numbers,
                                layer_alpha,
                                node_alpha,)
    # class_model.encoder = model.encoder

    # model = class_model
    # del class_model
    # model = model.cuda()

    if load_model:
        saved_path = './ckpts6/step_1600/model/model.safetensors'
        model.load_state_dict(load_file(saved_path))


    # 打印模型参数
    lst = []    
    for name, param in model.named_parameters():
        param.requires_grad = True
        if param.requires_grad == True:
            lst.append(param.nelement())
            # accelerator.print('name: ', name, 'param num: ', param.nelement())
    accelerator.print(f"trainable paras number: {sum(lst)}")

    # lora_layer_param = []
    # node_param = []
    # for name, param in model.named_parameters():
    #     param.requires_grad = True

    # optimizer = Adam(model.parameters(), lr=lr)
    optimizer = Adam([{'params': model.parameters(), 'lr': 1e-4}])

    # optimizer_2 = Adam([{'params': node_param, 'lr': 1e-5}])

    # 读取模型数据2
    # if load_model:
    #     logger.info(f"Loading trained model :./ckpts" + model_dirname + f"{saved_ep}/")
    #     ckpt = torch.load('./ckpts' + model_dirname + f"{saved_ep}/model.pt", weights_only=False)
    #     model.load_state_dict(ckpt['model_state'], strict=False)


    return model, optimizer

def Accuracy(y_true, y_pred):
    count = 0
    for i in range(y_true.size(0)):
        p = sum(torch.logical_and(y_true[i], y_pred[i]))
        q = sum(torch.logical_or(y_true[i], y_pred[i]))
        count += p / q
    return count


def get_col(matrix):
    return [matrix[:, col] for col in range(matrix.shape[1])]


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    labels = get_col(labels)
    preds = get_col(preds)
    auclist = []
    for idx in range(len(labels)):
        fpr, tpr, _ = roc_curve(labels[idx].flatten(), preds[idx].flatten())
        roc_auc = auc(fpr, tpr)
        auclist.append(roc_auc)
    return sum(auclist) / len(auclist)


def jaccard_similarity(a,b):
    intersection = (a * b).sum(dim=1)  # 计算交集
    union = (a + b - a * b).sum(dim=1)  # 计算并集
    jaccard = intersection / union  # 计算 Jaccard 相似度
    return jaccard


def metrics_compute(a,b):
    result_dict = {}
    for key in a:
        denominator = b.get(key, 0)
        if denominator == 0:
            result_dict[key] = 0
        else:
            result_dict[key] = round(a[key] / denominator, 3)
    return result_dict


def evaluate(model, validloader, accelerator: Accelerator):
    model.eval()
    subset_acc_num_layer0 = 0
    # subset_acc_num_layer1 = 0

    tp_sum_layer0 = 0
    pred_sum_layer0 = 0
    true_sum_layer0 = 0
    layer_batch_acc = 0
    # validlabels = []
    # predlabels = []
    tp_sum_layer1 = {}
    pred_sum_layer1 = {}
    true_sum_layer1 = {}
    with torch.inference_mode():
        for batch in validloader:
            output = model(**batch)
            global_logits, local_logits, layer_logits = model.layer_classifier(output.hidden_states)

            node_logits = model.node_classifier(layer_logits, output.hidden_states.detach())
            # predlabels.append(torch.sigmoid(node_logits.to(torch.float32)).cpu().numpy())
            # validlabels.append(refs0.numpy())

            # 计算样本的 subset acc
            if accelerator.sync_gradients:
                pred0 = (node_logits > 0).int()
                pred0, refs0 = accelerator.gather_for_metrics((pred0, batch["nodes"]))
                pred0 = pred0.cpu()
                refs0 = refs0.cpu()
                if accelerator.is_local_main_process:
                    subset_acc_num_layer0 += ((pred0 - refs0) == 0).all(dim=-1).sum().item()
                    # MCM0 = calculate_multilabel_confusion_matrix(pred0, refs0)
                    # tp_sum_layer0 += MCM0[:, 1, 1]
                    # pred_sum_layer0 += MCM0[:, 1, 1] + MCM0[:, 0, 1]
                    # true_sum_layer0 += MCM0[:, 1, 1] + MCM0[:, 1, 0]
                    # for go in high_freq:
                    #     idx = go2index(go)
                    #     if go not in tp_sum_layer1:
                    #         tp_sum_layer1[go] = 0
                    #         pred_sum_layer1[go] = 0
                    #         true_sum_layer1[go] = 0
                    #     tp_sum_layer1[go] += MCM0[idx, 1, 1]
                    #     pred_sum_layer1[go] += MCM0[idx, 1, 1] + MCM0[idx, 0, 1]
                    #     true_sum_layer1[go] += MCM0[idx, 1, 1] + MCM0[idx, 1, 0]

                # roc_auc = compute_roc(np.concatenate(validlabels,axis=0), np.concatenate(predlabels,axis=0))
                # accelerator.print('roc_auc:', roc_auc)

                pred1 = (layer_logits > 0).int()
                pred1, refs1 = accelerator.gather_for_metrics((pred1, batch["layers"]))
                pred1 = pred1.cpu()
                refs1 = refs1.cpu()
                layer_batch_acc += ((pred1 - refs1) == 0).all(dim=-1).sum().item()
        
        # precision = metrics_compute(tp_sum_layer1, pred_sum_layer1)
        # recall = metrics_compute(tp_sum_layer1, true_sum_layer1)

        # accelerator.print(f'频率大于{2e4}的验证集 precision:', precision)
        # accelerator.print(f'频率大于{2e4}的验证集 recall:', recall)
        accelerator.print(f'valid layer_batch_acc:', layer_batch_acc / len(validloader.dataset))
        # accelerator.print('验证集 TP / TP + FP:', tp_sum_layer1[0], ' / ', pred_sum_layer1[0])
        # accelerator.print('验证集 TP / TP + FN:', tp_sum_layer1[0], ' / ', true_sum_layer1[0])

    return (subset_acc_num_layer0 / len(validloader.dataset)
    )


def train(model, optimizer, accelerator: Accelerator, trainloader, validloader, epoch=epoch, log_step=10, resume=None):

    global_step = 0
    start_time = time.time()
    
    resume_step = 0
    resume_epoch = 0

    if resume is not None:
        accelerator.load_state(resume)
        # steps_per_epoch = math.ceil(len(trainloader) / accelerator.gradient_accumulation_steps)
        # resume_step = global_step = int(resume.split("step_")[-1])
        # resume_epoch = resume_step // steps_per_epoch
        # resume_step -= resume_epoch * steps_per_epochd
        accelerator.print(f"resume from checkpoint -> {resume}")

    use_mlp = False
    for ep in range(resume_epoch, epoch):
        model.train()
        subset_acc_num_layer0 = 0
        
        if resume and ep == resume_epoch and resume_step != 0:
            active_dataloader = accelerator.skip_first_batches(trainloader, resume_step * accelerator.gradient_accumulation_steps)
        else:
            active_dataloader = trainloader

        # active_dataloader.set_epoch(ep)
        # accelerator.print('dataloader had set epoch')

        for batch in active_dataloader:

            jac_sim = []
            with accelerator.accumulate(model):
                output = model(**batch)
                # layer_loss = output.layer_loss
                # node_loss = output.node_loss
                # regular_loss = output.regular_loss
                # loss = layer_loss + node_loss
                # accelerator.backward(loss)
                # optimizer.step()
                # optimizer.zero_grad()

                # layer_loss = output.layer_loss
                if not use_mlp:
                    global_logits, local_logits, layer_logits = model.layer_classifier(output.hidden_states)
                    
                    contra_loss = None

                    layer_loss, bottom_loss = LayerLoss(model.layer_loss_fn, 
                                                        global_logits, 
                                                        local_logits, 
                                                        batch["layers"], 
                                                        cluster_relations, 
                                                        model.penalty, 
                                                        is_multiss=True, 
                                                        use_hierar=True)

                    node_logits = model.node_classifier(layer_logits, output.hidden_states)

                    # 增加预测后处理
                    for col, rows in model.hierar_relations.items():
                        if len(rows) == 0:
                            continue
                        rows_with_col = torch.tensor([col] + rows, device=node_logits.device)
                        max_val, _ = node_logits[:,rows_with_col].max(dim=1)
                        node_logits[:, col] = max_val
                    
                    node_loss = NodeLoss(model.nodes_loss_fn, 
                                        node_logits, 
                                        batch["nodes"], 
                                        layer_loss,
                                        bottom_loss, 
                                        cluster_nodes_relations, 
                                        is_multi=True, 
                                        use_hierar=True,
                                        hierar_relation=model.hierar_relations
                            )

                # node_loss = torch.nn.BCEWithLogitsLoss()(node_logits,batch["nodes"])
                
                # accelerator.print('layer pos loss sum:', round(layer_loss[batch["layers"][:,:len(cluster_relations.keys())] == 1].mean().item(),2),
                #                   'layer neg loss sum:', round(layer_loss[batch["layers"][:,:len(cluster_relations.keys())] == 0].mean().item(),2),
                #                   'bottom pos loss sum:', round(bottom_loss[batch["layers"][:,len(cluster_relations.keys()):] == 1].mean().item(),2),
                #                   'bottom neg loss sum:', round(bottom_loss[batch["layers"][:,len(cluster_relations.keys()):] == 0].mean().item(),2),
                #                   'node pos loss sum:', round(node_loss[batch["nodes"] == 1].mean().item(),2),
                #                   'node neg loss sum:', round(node_loss[batch["nodes"] == 0].mean().item(),2))
                
                    layer_loss = layer_loss.mean()
                    bottom_loss = bottom_loss.mean()
                    node_loss = node_loss.mean()
                    loss = node_loss
                else:
                    node_logits = model.mlp(output.hidden_states)
                    loss = torch.nn.BCEWithLogitsLoss(reduction='none')(node_logits, batch["nodes"]).mean()

                # layer_loss = output.layer_loss
                optimizer.zero_grad()
                accelerator.backward(loss) # , retain_graph=True
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                

                # accelerator.backward(node_loss)
                # optimizer2.step()
                # optimizer2.zero_grad()

                # 计算样本的 subset acc
                # if accelerator.sync_gradients:
                pred0 = (node_logits > 0).int()
                pred0, refs0 = accelerator.gather_for_metrics((pred0, batch["nodes"]))
                pred0 = pred0.cpu()
                refs0 = refs0.cpu()
                subset_acc_num_layer0 += ((pred0 - refs0) == 0).all(dim=-1).sum().item()
                jac_sim.append(jaccard_similarity(pred0,refs0))
                
                # pred1 = (layer_logits > 0).int()
                # pred1, refs1 = accelerator.gather_for_metrics((pred1, batch["layers"]))
                # pred1 = pred1.cpu()
                # refs1 = refs1.cpu()

                # layer_batch_acc = ((refs1 - pred1) == 0).all(dim=-1).sum().item() / pred1.size()[0]
            
                global_step += 1

                if global_step % log_step == 0:
                    loss = accelerator.reduce(loss, "mean")
                    # layer_loss = accelerator.reduce(layer_loss, "mean")
                    # bottom_loss = accelerator.reduce(bottom_loss, "mean")
                    # node_loss = accelerator.reduce(node_loss, "mean")
                    # regular_loss = accelerator.reduce(regular_loss, "mean")
                    accelerator.print(  f"ep: {ep}, "
                                        f"global_step: {global_step}, "
                                        # f"layer accuracy: {layer_batch_acc}, "
                                        f"total loss: {loss.item()}, "
                                        # f"layer loss: {layer_loss.item()}, "
                                        # f"bottom loss: {bottom_loss.item()}, "
                                        # f"node loss: {node_loss.item()}"
                                    )

                    # accelerator.print(f"jac_sim ave:{torch.concat(jac_sim).mean()}")
                    # accelerator.log({"total_loss": loss.item()}, global_step)
                    # accelerator.log({"loss": flood.item()}, global_step)

                if global_step % 500 == 0 and global_step != 0:
                    accelerator.print(f"save checkpoint -> step_{global_step}")
                    accelerator.save_state(output_dir=accelerator.project_dir + f"/step_{global_step}")

                    if accelerator.is_local_main_process:
                        
                        # unwrap_model = accelerator.unwrap_model(model)      

                        # if not os.path.exists('./ckpts' + model_dirname + f"{ep}/"):
                        #     os.makedirs('./ckpts' + model_dirname + f"{ep}/")

                        # accelerator.save_state(output_dir=accelerator.project_dir + f"/step_{ep}")
                        accelerator.unwrap_model(model).save_pretrained(
                                save_directory=accelerator.project_dir + f"/step_{global_step}/model",
                                # is_main_process=accelerator.is_main_process,
                                state_dict=accelerator.get_state_dict(model),
                                save_func=accelerator.save
                                )
                        # torch.save({
                        #     'model_state' : unwrap_model.state_dict()}, 
                        #     './ckpts' + model_dirname + f"{ep}/model.pt")
                        
                        logger.info(f'step_{global_step}.savetensors is saved...')
                        accelerator.print(f"step_{global_step} saved!")

        # valid_acc0 = evaluate(model, validloader, accelerator)
        # accelerator.print(f"ep: {ep}, time: {time.time() - start_time}")

        # accelerator.print('训练集的acc:', subset_acc_num_layer0 / len(trainloader.dataset))
        # accelerator.log({"训练集的acc": subset_acc_num_layer0 / len(trainloader.dataset)}, global_step)
        # accelerator.print('验证集的acc:', valid_acc0)
        # accelerator.log({"验证集的acc": valid_acc0}, global_step)

        # accelerator.print('valid auc:', roc_auc)

    accelerator.end_training()

def main() -> None:

    deepspeed_plugin = DeepSpeedPlugin()
    deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = train_batch_size
    accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin,
                              log_with="tensorboard", 
                              project_dir="ckpts4")

    accelerator.init_trackers("runs")

    resume = None
    resume = './ckpts4/step_2000'

    trainloader, validloader = prepare_dataloader(accelerator)
    model, optimizer = prepare_model_and_optimizer(accelerator)
    model, optimizer, trainloader, validloader = accelerator.prepare(model, optimizer, trainloader, validloader)
    train(model, optimizer, accelerator, trainloader, validloader, resume=resume)

    # train(model, optimizer, trainloader, validloader, accelerator, resume='./ckpts2/step_2400')

    # evaluate(model, validloader, accelerator) 

if __name__ == "__main__":
    main()