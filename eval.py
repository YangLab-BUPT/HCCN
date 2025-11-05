import pandas as pd
import torch
import ast
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import AveragePrecision
from joblib import Parallel, delayed
import json

root = '/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/'
onto = 'bpo'
data_root = root + onto
path = data_root + '/withlayer/stride/'


test_data_file = path + 'testdata.csv'
col = list(pd.read_csv(test_data_file,nrows=0))[2:]
index2go = {}
for i in range(len(col)):
    index2go[i] = col[i]

predlabels = np.load(f'/data/csyData/uniprot_test/code/GOcode/{onto}_version2/result/predlabels0_mlp.npy')
# pvalidlabels = np.load(f'/data/csyData/uniprot_test/code/GOcode/{onto}_version2/result/pvalidlabels0.npy')

with open(f'/data/csyData/uniprot_test/code/GOcode/{onto}_version2/freq.json') as f:
    label_freq = json.load(f)
choose_col = []
reject_col = []
for key, value in label_freq.items():
    if value < 0.8:
        choose_col.append(col.index(key))
    else:
        reject_col.append(col.index(key))


# test_blast_data_file = path + 'testdata_blast2.pkl'
# test_blast_data = pd.read_pickle(test_blast_data_file)
        
test_diamond_data_file = path + 'testdata_diamond.pkl'
test_diamond_data = pd.read_pickle(test_diamond_data_file)

# true_labels_diamond = np.array(test_blast_data['true_labels'].to_list()) 
# blast_preds_diamond = np.array(test_blast_data['blast_preds'].to_list())
# pvalidlabels = np.array(test_blast_data['true_labels'].to_list())

blast_preds_diamond = np.array(test_diamond_data['blast_preds'].to_list())
pvalidlabels = np.array(test_diamond_data['true_labels'].to_list())

blast_labels = blast_preds_diamond[:,choose_col]
my_labels = predlabels[:,choose_col]
true_labels = pvalidlabels[:,choose_col]
# blast_preds = predlabels[:,choose_col]

def compute_metrics(label_idx):
    precision, recall, _ = precision_recall_curve(true_labels[:, label_idx], my_labels[:, label_idx])
    auprc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(true_labels[:, label_idx], my_labels[:, label_idx])
    auroc = auc(fpr, tpr)
    return auprc, auroc

def compute_diamond_metrics(label_idx):
    precision, recall, _ = precision_recall_curve(true_labels[:, label_idx], blast_labels[:, label_idx])
    auprc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(true_labels[:, label_idx], blast_labels[:, label_idx])
    auroc = auc(fpr, tpr)
    return auprc, auroc


results = Parallel(n_jobs=-1)(
    delayed(compute_metrics)(i) for i in range(my_labels.shape[1])
)


auprc_results = [result[0] for result in results]
auroc_results = [result[1] for result in results]

result2 = Parallel(n_jobs=-1)(
    delayed(compute_diamond_metrics)(i) for i in range(blast_labels.shape[1])
)
auprc_results2 = [result[0] for result in result2]
auroc_results2 = [result[1] for result in result2]

better = {}
worse = {}
for idx, i in enumerate(choose_col):
    if auprc_results[idx] > auroc_results2[idx]:
        better[col[i]] = {"blast": auroc_results2[idx],"my_pred":auprc_results[idx]}
    else:
        worse[col[i]] = {"blast": auroc_results2[idx],"my_pred":auprc_results[idx]}

with open(f'{onto}_high_freq_better_diamond_mlp.json','w') as f:
    json.dump(better,f,indent=4)
with open(f'{onto}_high_freq_worse_diamond_mlp.json','w') as f:
    json.dump(worse,f,indent=4)



blast_labels = blast_preds_diamond[:,reject_col]
true_labels = pvalidlabels[:,reject_col]
my_labels = predlabels[:,reject_col]

def compute_metrics(label_idx):
    precision, recall, _ = precision_recall_curve(true_labels[:, label_idx], my_labels[:, label_idx])
    auprc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(true_labels[:, label_idx], my_labels[:, label_idx])
    auroc = auc(fpr, tpr)
    return auprc, auroc

def compute_diamond_metrics(label_idx):
    precision, recall, _ = precision_recall_curve(true_labels[:, label_idx], blast_labels[:, label_idx])
    auprc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(true_labels[:, label_idx], blast_labels[:, label_idx])
    auroc = auc(fpr, tpr)
    return auprc, auroc

results = Parallel(n_jobs=-1)(
    delayed(compute_metrics)(i) for i in range(my_labels.shape[1])
)


auprc_results = [result[0] for result in results]
auroc_results = [result[1] for result in results]

result2 = Parallel(n_jobs=-1)(
    delayed(compute_diamond_metrics)(i) for i in range(blast_labels.shape[1])
)
auprc_results2 = [result[0] for result in result2]
auroc_results2 = [result[1] for result in result2]

better = {}
worse = {}
for idx, i in enumerate(reject_col):
    if auprc_results[idx] > auroc_results2[idx]:
        better[col[i]] = {"blast": auroc_results2[idx],"my_pred":auprc_results[idx]}
    else:
        worse[col[i]] = {"blast": auroc_results2[idx],"my_pred":auprc_results[idx]}

with open(f'{onto}_low_freq_better_diamond_mlp.json','w') as f:
    json.dump(better,f)
with open(f'{onto}_low_freq_worse_diamond_mlp.json','w') as f:
    json.dump(worse,f)


# sorted_data_list = sorted(result, reverse=True)
# bin_edges = list(np.arange(0.4, 0.8, 0.05))
# counts, bins = np.histogram(result, bins=bin_edges)
# bin_widths = np.diff(bin_edges)
# total_bar_width = 0.8
# bar_width_proportion = 0.9
# actual_bar_width = total_bar_width * bar_width_proportion
# spacing_width = total_bar_width - actual_bar_width
# num_bins = len(bins) - 1
# bar_positions = np.arange(num_bins) + np.arange(num_bins) * spacing_width
# plt.figure(figsize=(12, 6))
# bars = plt.bar(bar_positions, counts, width=actual_bar_width, align='edge', edgecolor='black', color='skyblue')
# bin_labels = [f"{round(start,2)}-{round(end,2)}" for start, end in zip(bins[:-1], bins[1:])]
# plt.xticks(bar_positions + actual_bar_width / 2, bin_labels, rotation=45, ha='right')
# plt.title(f'Variety of Labels Across Different Frequency Ranges After Cluster')
# plt.xlabel(f'the auprc of {onto}')
# plt.ylabel('Number of Unique Label')
# plt.tight_layout()
# plt.savefig(f'{onto} auprc.png', bbox_inches='tight')


# # blast_preds = torch.tensor(test_blast_data['blast_preds'].to_list())
# # true_labels = torch.tensor(test_blast_data['true_labels'].to_list())

# def calculate_multilabel_confusion_matrix(y_true, y_pred):

#     y_true = y_true.long()
#     y_pred = (y_pred > 0.5).long()

#     true_positives = (y_true == 1) & (y_pred == 1)
#     false_positives = (y_true == 0) & (y_pred == 1)
#     true_negatives = (y_true == 0) & (y_pred == 0)
#     false_negatives = (y_true == 1) & (y_pred == 0)

#     tp_counts = torch.sum(true_positives, dim=0)
#     fp_counts = torch.sum(false_positives, dim=0)
#     tn_counts = torch.sum(true_negatives, dim=0)
#     fn_counts = torch.sum(false_negatives, dim=0)

#     # p1 = (tp_counts / (tp_counts + fp_counts))[(tp_counts / (tp_counts + fp_counts)) > 0]
#     # r1 = (tp_counts / (tp_counts + fn_counts))[(tp_counts / (tp_counts + fn_counts)) > 0]
#     for i in range(tp_counts.size()[0]):
#         p1 = (tp_counts[i] / (tp_counts[i] + fp_counts[i]))
#         r1 = (tp_counts[i] / (tp_counts[i] + fn_counts[i]))
#         if r1 > 0:
#             print(f"{index2go[i]} : precision = {p1}, recall = {r1}")

#     # confusion_matrices = torch.stack([
#     #     torch.stack([tn_counts, fp_counts], dim=-1),
#     #     torch.stack([fn_counts, tp_counts], dim=-1)
#     # ], dim=-2)

#     # return [cm for cm in confusion_matrices.unbind(0)]
    

# # # ((y_true == y_pred).all(dim=1)).sum().item()
# # calculate_multilabel_confusion_matrix(true_labels, blast_preds)

