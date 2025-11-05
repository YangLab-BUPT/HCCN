#!/usr/bin/env python
import click as ck
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import logging, math, time
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from aminoacids import to_ngrams, to_onehot, MAXLEN

import json
from sklearn.metrics import roc_auc_score
import numpy as np


class MultiScaleConv1D(nn.Module):
    def __init__(self, num_classes=5772):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        kernel_sizes = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
        for kernel_size in kernel_sizes:
            conv_layer = nn.Conv1d(
                in_channels=21,
                out_channels=512,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=True
            )
            self.conv_layers.append(conv_layer)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(8192, num_classes)  # 8192 = 16 * 512
        self.activation = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_out = self.activation(conv_layer(x))
            pooled = self.global_pool(conv_out).view(batch_size, -1)
            conv_outputs.append(pooled)
        combined_features = torch.cat(conv_outputs, dim=1)
        return self.output_layer(combined_features)



class DeepGOPDataset(Dataset):
    def __init__(self, df, terms_dict, nb_classes):
        self.df = df.reset_index(drop=True)
        self.terms_dict = terms_dict
        self.nb_classes = nb_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row.sequences
        onehot = to_onehot(seq).astype(np.float32)
        labels = np.zeros((self.nb_classes,), dtype=np.float32)
        for t_id in row.prop_annotations:
            if t_id in self.terms_dict:
                labels[self.terms_dict[t_id]] = 1.0
        return onehot, labels, idx


def collate_fn(batch):
    xs = np.stack([b[0] for b in batch], axis=0)
    ys = np.stack([b[1] for b in batch], axis=0)
    idxs = np.array([b[2] for b in batch], dtype=np.int64)
    xs = torch.from_numpy(xs).permute(0, 2, 1).contiguous()
    ys = torch.from_numpy(ys)
    idxs = torch.from_numpy(idxs)
    return xs, ys, idxs


import json
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def evaluate_per_label_auc_on_test(model_path, test_df, terms_dict, terms, nb_classes, device, batch_size=256, save_path="per_label_auc.json"):
    """
    对 test_df 推理并计算每个标签的 AUC（无法计算则为 0），结果保存为 JSON 文件
    """
    test_loader = DataLoader(
        DeepGOPDataset(test_df, terms_dict, nb_classes),
        batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=collate_fn, pin_memory=True
    )

    model = MultiScaleConv1D(num_classes=nb_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb, _ in tqdm(test_loader, desc="Predicting on test set"):
            xb = xb.to(device)
            probs = torch.sigmoid(model(xb)).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(yb.numpy())
    preds = np.vstack(all_preds)
    labels = np.vstack(all_labels)

    # === 计算每个标签的 AUC ===
    n_classes = labels.shape[1]
    aucs = {}
    for i in range(n_classes):
        y_true = labels[:, i]
        y_score = preds[:, i]
        if len(np.unique(y_true)) < 2:
            auc_value = 0.0
        else:
            try:
                auc_value = float(roc_auc_score(y_true, y_score))
            except ValueError:
                auc_value = 0.0
        aucs[terms[i]] = auc_value

    with open(save_path, "w") as f:
        json.dump(aucs, f, indent=2)

    print(f"✅ Per-label AUC saved to {save_path}")
    return aucs



def compute_per_label_auc(labels, preds):
    from sklearn.metrics import roc_auc_score

    n_classes = labels.shape[1]
    aucs = np.full(n_classes, np.nan)  # 先全部设为 NaN
    
    for i in range(n_classes):
        y_true = labels[:, i]
        y_score = preds[:, i]
        
        if len(np.unique(y_true)) < 2:
            continue
        try:
            aucs[i] = roc_auc_score(y_true, y_score)
        except ValueError:
            aucs[i] = np.nan
    
    return aucs


@ck.command()
@ck.option('--go-file', '-gf', default='/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/go-basic-2311.obo', help='Gene Ontology file in OBO Format')
@ck.option('--train-data-file', '-trdf', default='/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/cco/withlayer/stride/traindata.pkl', help='Data file with sequences and complete set of annotations')
@ck.option('--test-data-file', '-tsdf', default='/data/csyData/uniprot_test/data/XML_testdata/go_temp/GO_data/cco/withlayer/stride/testdata.pkl', help='Data file with sequences and complete set of annotations')
@ck.option('--terms-file', '-tf', default='data/mfo_terms.pkl', help='Data file with sequences and complete set of annotations')
@ck.option('--model-file', '-mf', default='data/deepgoplus_pytorch.pth', help='DeepGOPlus model file to save/load')
@ck.option('--out-file', '-o', default='data/predictions.pkl', help='Result file with predictions for test set')
@ck.option('--split', '-s', default=0.9, type=float, help='train/valid split')
@ck.option('--batch-size', '-bs', default=256, type=int, help='Batch size')
@ck.option('--epochs', '-e', default=20, type=int, help='Training epochs')
@ck.option('--load', '-ld', is_flag=True, help='Load Model?')
@ck.option('--logger-file', '-lf', default='data/training.csv', help='CSV logger file')
@ck.option('--threshold', '-th', default=0.5, type=float, help='Prediction threshold')
@ck.option('--device', '-d', default='cuda:0', help='Device e.g. cpu or cuda:0')
def main(go_file, train_data_file, test_data_file, terms_file, model_file,
         out_file, split, batch_size, epochs, load, logger_file, threshold,
         device):

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load data
    df = pd.read_pickle(train_data_file)
    test_df = pd.read_pickle(test_data_file)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    nb_classes = 5518

    """
    MFO 5518
    BPO 7962
    CCO 1556
    EC  4238

    """

    pretrained_file = '/data/csyData/uniprot_test/code/GOcode/upload/HCCN/HCCN/deepgoplus/data/deepgoplus_pytorch.pth'
    num_label = nb_classes
    output_dir = '/data/csyData/uniprot_test/code/GOcode/upload/HCCN/HCCN/deepgoplus/deepgoplus_output_cco'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_results = []

    train = True
    # Cross-validation
    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(df)):
        logging.info(f"===== Fold {fold_idx + 1}/10 =====")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)

        train_loader = DataLoader(DeepGOPDataset(train_df, terms_dict, nb_classes),
                                  batch_size=batch_size, shuffle=True, num_workers=4,
                                  collate_fn=collate_fn, pin_memory=True)
        valid_loader = DataLoader(DeepGOPDataset(valid_df, terms_dict, nb_classes),
                                  batch_size=batch_size, shuffle=False, num_workers=4,
                                  collate_fn=collate_fn, pin_memory=True)

        base_model = MultiScaleConv1D(num_classes=5772)
        ckpt = torch.load(pretrained_file, map_location=device)
        if 'model_state_dict' in ckpt:
            base_model.load_state_dict(ckpt['model_state_dict'], strict=False)
        else:
            base_model.load_state_dict(ckpt, strict=False)

        base_model.output_layer = nn.Linear(8192, num_label)
        for name, param in base_model.named_parameters():
            param.requires_grad = ('output_layer' in name)

        model = base_model.to(device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        best_model_path = Path(output_dir) / f"deepgoplus_fold{fold_idx + 1}.pth"
        log_path = Path(output_dir) / f"training_fold{fold_idx + 1}.csv"
        history = []

        if train:
            for epoch in range(1, epochs + 1):
                model.train()
                train_loss = 0.0
                n_train = 0
                for xb, yb, _ in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * xb.size(0)
                    n_train += xb.size(0)
                train_loss /= max(1, n_train)

                # validation
                model.eval()
                val_loss, n_val = 0.0, 0
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for xb, yb, _ in tqdm(valid_loader, desc=f"Fold {fold_idx+1} Epoch {epoch} [Valid]"):
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        val_loss += loss.item() * xb.size(0)
                        n_val += xb.size(0)
                        all_preds.append(torch.sigmoid(logits).cpu().numpy())
                        all_labels.append(yb.cpu().numpy())

                val_loss /= max(1, n_val)
                preds = np.vstack(all_preds)
                labels = np.vstack(all_labels)
                # roc_auc = compute_multilabel_auc(labels, preds)
                print(f"Fold {fold_idx+1} Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
                # print(f"Fold {fold_idx+1} Epoch {epoch}: ROC AUC: Micro={roc_auc['micro_auc']:.4f}, Macro={roc_auc['macro_auc']:.4f}, Samples={roc_auc['samples_auc']:.4f}")
                history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
                pd.DataFrame(history).to_csv(log_path, index=False)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), best_model_path)
                    logging.info(f"Fold {fold_idx+1}: Improved val_loss={val_loss:.4f}")
                    logging.info(f"Fold {fold_idx+1}: Model saved to {best_model_path}")
                else:
                    logging.info(f"Fold {fold_idx+1}: No improvement (val_loss={val_loss:.4f})")

        # model.load_state_dict(torch.load(best_model_path, map_location=device))
        # model.eval()
        # all_preds, all_labels = [], []
        # with torch.no_grad():
        #     for xb, yb, _ in DataLoader(DeepGOPDataset(valid_df, terms_dict, nb_classes),
        #                                 batch_size=batch_size, shuffle=False, num_workers=4,
        #                                 collate_fn=collate_fn):
        #         xb = xb.to(device)
        #         logits = model(xb)
        #         probs = torch.sigmoid(logits).cpu().numpy()
        #         all_preds.append(probs)
        #         all_labels.append(yb.numpy())
        # preds = np.vstack(all_preds)
        # labels = np.vstack(all_labels)

        # test_model_path = Path(output_dir) / f"deepgoplus_fold{fold_idx+1}.pth"
        # auc_json_path = Path(output_dir) / f"test_per_label_auc{fold_idx+1}.json"
        # evaluate_per_label_auc_on_test(
        #     model_path=test_model_path,
        #     test_df=test_df,
        #     terms_dict=terms_dict,
        #     terms=terms,
        #     nb_classes=nb_classes,
        #     device=device,
        #     batch_size=batch_size,
        #     save_path=auc_json_path
        # )


    mean_auc = np.mean(fold_results)
    logging.info(f"10-Fold Mean ROC AUC: {mean_auc:.4f}")


if __name__ == '__main__':
    main()
