#!/usr/bin/env python
# coding:utf-8
"""
Tencent is pleased to support the open source community by making NeuralClassifier available.
Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the MIT License (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at
http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for thespecific language governing permissions and limitations under
the License.
"""

import torch
import torch.nn as nn
import json
from util import Type
import torch.nn.functional as F

class LossType(Type):
    """Standard names for loss type
    """
    SOFTMAX_CROSS_ENTROPY = "SoftmaxCrossEntropy"
    SOFTMAX_FOCAL_CROSS_ENTROPY = "SoftmaxFocalCrossEntropy"
    SIGMOID_FOCAL_CROSS_ENTROPY = "SigmoidFocalCrossEntropy"
    BCE_WITH_LOGITS = "BCEWithLogitsLoss"
    BCE_WITH_LOGITS2 = "BCEWithLogitsLoss2"

    @classmethod
    def str(cls):
        return ",".join([cls.SOFTMAX_CROSS_ENTROPY,
                         cls.SOFTMAX_FOCAL_CROSS_ENTROPY,
                         cls.SIGMOID_FOCAL_CROSS_ENTROPY,
                         cls.BCE_WITH_LOGITS])


class ActivationType(Type):
    """Standard names for activation type
    """
    SOFTMAX = "Softmax"
    SIGMOID = "Sigmoid"

    @classmethod
    def str(cls):
        return ",".join([cls.SOFTMAX,
                         cls.SIGMOID])


class FocalLoss(nn.Module):
    """Softmax focal loss
    references: Focal Loss for Dense Object Detection
                https://github.com/Hsuxu/FocalLoss-PyTorch
    """

    def __init__(self, label_size, activation_type=ActivationType.SOFTMAX, alpha=0.25,
                 gamma=2.0, epsilon=1.e-9):
        super(FocalLoss, self).__init__()
        self.num_cls = label_size
        self.activation_type = activation_type
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = 1.0


    def forward(self, logits, target):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        if self.activation_type == ActivationType.SOFTMAX:
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_cls,
                                      dtype=torch.float,
                                      device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            logits = torch.softmax(logits, dim=-1)
            loss = -self.alpha * one_hot_key * \
                   torch.pow((1 - logits), self.gamma) * \
                   (logits + self.epsilon).log()
            loss = loss.sum(1)
        elif self.activation_type == ActivationType.SIGMOID:
            
            # 法3
            probas = torch.sigmoid(logits)
            pt = probas.where(target == 1, 1 - probas)  # 如果目标是 1，则取 probas；如果目标是 0，则取 1 - probas
            
            alpha = torch.tensor(self.alpha).view(1, -1).expand_as(target).to(pt.device)  # 扩展 alpha 以匹配 targets 的形状
            loss = -alpha * (1 - pt) ** self.gamma * torch.log(pt + self.epsilon)

            # 法1
            # multi_hot_key = target
            # logits = torch.sigmoid(logits)
            # zero_hot_key = 1 - multi_hot_key
            # alpha = torch.tensor(self.alpha).expand_as(target)

            # loss = - alpha.to(multi_hot_key.device) * multi_hot_key * \
            #        torch.pow((1 - logits), self.gamma) * \
            #        (logits + self.epsilon).log()
            # loss += -(1 - alpha.to(multi_hot_key.device)) * zero_hot_key * \
            #         torch.pow(logits, self.gamma) * \
            #         (1 - logits + self.epsilon).log()
            
            # 法2
            # bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
            # alpha = torch.tensor(self.alpha).to(target.device) * target + (1 - torch.tensor(self.alpha).to(target.device)) * (1 - target)
            # loss = (alpha * ((1 - logits) ** self.gamma) * bce_loss).mean()

            # seesaw loss
            # probs = torch.sigmoid(logits)
            # positive_counts = target.sum(dim=0)
            # total_samples = target.size(0)
            # positive_ratios = positive_counts / total_samples
            # avg_positive_ratio = positive_ratios.mean()
            # lambda_weights = 1 / (1 + torch.exp(-self.beta * (positive_ratios - avg_positive_ratio)))
            # ce_loss = F.binary_cross_entropy(probs, target, reduction='none')
            # focal_term = (1 - probs) ** self.gamma * ce_loss
            # loss = lambda_weights * focal_term * target + (1 - lambda_weights) * focal_term * (1 - target)

        else:
            raise TypeError("Unknown activation type: " + self.activation_type
                            + "Supported activation types: " +
                            ActivationType.str())
        return loss.mean()


class ClassificationLoss(torch.nn.Module):
    def __init__(self, label_size, layer_alpha, node_alpha, class_weight=None,
                 loss_type=LossType.SOFTMAX_CROSS_ENTROPY,
                 
                 ):
        super(ClassificationLoss, self).__init__()
        self.label_size = label_size
        self.loss_type = loss_type

        self.gamma_pos = 0
        self.gamma_neg = 4
        self.clip = 0.20
        self.node_gamma_neg = 4
        self.node_clip = 0.2
        self.layer_alpha = torch.tensor(layer_alpha, dtype=torch.bfloat16).cuda()
        self.node_alpha = torch.tensor(node_alpha, dtype=torch.bfloat16).cuda()

        if loss_type == LossType.SOFTMAX_CROSS_ENTROPY:
            self.criterion = torch.nn.CrossEntropyLoss(class_weight)
        elif loss_type == LossType.SOFTMAX_FOCAL_CROSS_ENTROPY:
            self.criterion = FocalLoss(label_size, ActivationType.SOFTMAX, layer_alpha)
        elif loss_type == LossType.SIGMOID_FOCAL_CROSS_ENTROPY:
            self.criterion = FocalLoss(label_size, ActivationType.SIGMOID, layer_alpha)
        elif loss_type == LossType.BCE_WITH_LOGITS:
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise TypeError(
                "Unsupported loss type: %s. Supported loss type is: %s" % (
                    loss_type, LossType.str()))

    def forward(self, logits, target,
                use_hierar=False,
                is_multi=False,
                *argvs):
        device = logits.device
        if use_hierar:
            if self.loss_type == LossType.BCE_WITH_LOGITS:
                _, _, hierar_relations = argvs[0:3]
                all_loss = self.criterion(logits, target)
                
                # copy from https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
                xs_pos = torch.sigmoid(logits)
                xs_neg = 1 - xs_pos
                xs_neg = (xs_neg + self.clip).clamp(max=1)
                pt0 = xs_pos * target
                pt1 = xs_neg * (1 - target)  # pt = p if t > 0 else 1-p
                pt = pt0 + pt1
                one_sided_gamma = self.gamma_pos * target + self.node_gamma_neg * (1 - target)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
                all_loss *= one_sided_w

                layer_loss = all_loss[:,:len(hierar_relations.keys())]

                return layer_loss, self.cal_recursive_regularize3(all_loss,
                                                                    hierar_relations,
                                                                    device)

            elif self.loss_type == LossType.BCE_WITH_LOGITS2:
                criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
                _, layer_loss, bottom_loss, cluster_nodes_relations, hierar_relations = argvs[0:5]

                all_loss = criterion(logits, target)

                xs_pos = torch.sigmoid(logits)
                xs_neg = 1 - xs_pos
                xs_neg = (xs_neg + self.node_clip).clamp(max=1)
                pt0 = xs_pos * target
                pt1 = xs_neg * (1 - target)  # pt = p if t > 0 else 1-p
                pt = pt0 + pt1
                one_sided_gamma = self.gamma_pos * target + self.gamma_neg * (1 - target)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
                all_loss *= one_sided_w

                return all_loss + 0.5 * self.cal_recursive_regularize2(all_loss,
                                                                    layer_loss,
                                                                    bottom_loss,
                                                                    cluster_nodes_relations,
                                                                    device)

        else:
            if is_multi:
                assert self.loss_type in [LossType.BCE_WITH_LOGITS,
                                          LossType.SIGMOID_FOCAL_CROSS_ENTROPY]
            else:
                if self.loss_type not in [LossType.SOFTMAX_CROSS_ENTROPY,
                                          LossType.SOFTMAX_FOCAL_CROSS_ENTROPY]:
                    pass
            return self.criterion(logits, target)

    def cal_recursive_regularize(self, paras, hierar_relations, device="cpu"):
        """ Only support hierarchical text classification with BCELoss
        references: http://www.cse.ust.hk/~yqsong/papers/2018-WWW-Text-GraphCNN.pdf
                    http://www.cs.cmu.edu/~sgopal1/papers/KDD13.pdf
        """
        recursive_loss = 0.0
        for i in hierar_relations.keys():
            children_ids = hierar_relations[i]
            children_ids_list = torch.tensor(children_ids, dtype=torch.long).to(paras.device)
            children_paras = torch.index_select(paras, 1, children_ids_list)
            parent_para = torch.index_select(paras, 1, torch.tensor(int(i)).to(paras.device))
            parent_para = parent_para.repeat(1,children_ids_list.size()[0])
            diff_paras = parent_para - children_paras
            diff_paras = torch.relu(diff_paras.view(diff_paras.size()[0], -1))
            recursive_loss += 1.0 / 2 * torch.norm(diff_paras, p=2) ** 2
        return recursive_loss
    
    def cal_recursive_regularize2(self, all_loss, layer_loss, bottom_loss, hierar_relations, device="cpu"):
        """ Only support hierarchical text classification with BCELoss
        references: http://www.cse.ust.hk/~yqsong/papers/2018-WWW-Text-GraphCNN.pdf
                    http://www.cs.cmu.edu/~sgopal1/papers/KDD13.pdf
        """
        recursive_loss = []
        for i in range(bottom_loss.size(1)):  # 只拿第二层的计算损失
            index = i + layer_loss.size(1)
            children_ids = hierar_relations[str(index)]
            if not children_ids:
                continue
            parent_loss = all_loss[:, [int(i)]]
            children_loss = all_loss[:, children_ids]
            adjusted_children = torch.maximum(parent_loss, children_loss)
            all_loss[:, children_ids] = adjusted_children
            recursive_loss.append(adjusted_children)
        return torch.concat(recursive_loss,dim=1)
    
    def cal_recursive_regularize3(self, all_loss, hierar_relations, device):
        recursive_loss = []
        for i in hierar_relations.keys():
            children_ids = hierar_relations[i]
            if not children_ids:
                continue
            parent_loss = all_loss[:, [int(i)]]
            children_loss = all_loss[:, children_ids]
            adjusted_children = torch.maximum(parent_loss, children_loss)
            all_loss[:, children_ids] = adjusted_children
            recursive_loss.append(adjusted_children)
        return torch.cat(recursive_loss, dim=1)