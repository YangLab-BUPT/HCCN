import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from transformers import set_seed
from loss import ClassificationLoss
from loss import LossType
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.utils import ModelOutput
import heapq
import os


# 设置种子，保证实验可再现
def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)


#****参考的源地址：https://gist.github.com/sam-writer/723baf81c501d9d24c6955f201d86bbb
#****以及 https://github.com/agemagician/ProtTrans/blob/master/Fine-Tuning/PT5_LoRA_Finetuning_per_prot.ipynb
class ClassConfig:
    def __init__(self, dropout=0.15, node_nums=1, layer_nums=1, batch_size=8):
        self.dropout_rate = dropout
        self.node_nums = node_nums
        self.layer_nums = layer_nums
        self.batch_size = batch_size

class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(SelfAttentionLayer, self).__init__()
        self.query = nn.Linear(hidden_size, output_size)
        self.key = nn.Linear(hidden_size, output_size)
        self.value = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)
        self.scale_factor = torch.sqrt(torch.tensor(output_size, dtype=torch.float32))

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attn_weights = torch.matmul(Q, K.permute(0,1)) / self.scale_factor
        attn_weights = self.softmax(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        return attn_output


# ckpts2: HMCN2 + Head
class HMCNClassificationHead2(nn.Module):
    def __init__(self, config, class_config):
        super().__init__()
        self.hierarchical_depth = [0, 4096, 4096]
        self.global2local = [0, 4096, 4096]
        self.hierarchical_class = [48, 584]

        self.local_layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()

        for i in range(1, len(self.hierarchical_depth)):

            global_layer = nn.Sequential(
                nn.Linear(config.hidden_size + self.hierarchical_depth[i-1], self.hierarchical_depth[i]),
                nn.ReLU(),
                nn.BatchNorm1d(self.hierarchical_depth[i]),
                nn.Dropout(p=0.1),
                nn.Linear(self.hierarchical_depth[i], self.hierarchical_depth[i]),
                nn.ReLU(),
                nn.BatchNorm1d(self.hierarchical_depth[i]),
                ResidualBlock(self.hierarchical_depth[i],self.hierarchical_depth[i]),
                nn.Linear(self.hierarchical_depth[i], self.hierarchical_depth[i]),
                ResidualBlock(self.hierarchical_depth[i],self.hierarchical_depth[i]),
            )
            self.global_layers.append(global_layer)

            local_layer = nn.Sequential(
                nn.Linear(self.hierarchical_depth[i], self.global2local[i]), 
                nn.ReLU(),
                nn.BatchNorm1d(self.global2local[i]),
                nn.Dropout(p=0.1),
                nn.Linear(self.global2local[i], self.global2local[i]),
                nn.ReLU(),
                nn.BatchNorm1d(self.global2local[i]), 
                nn.Linear(self.global2local[i], self.hierarchical_class[i-1])
            )
            self.local_layers.append(local_layer)

        self.global_layers.apply(self._init_weight)
        self.local_layers.apply(self._init_weight)

        self.linear = nn.Sequential(
            nn.Linear(self.hierarchical_depth[-1], 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            ResidualBlock(2048,2048),
            ResidualBlock(2048,2048),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, class_config.layer_nums)
        )
        self.linear.apply(self._init_weight)

    def _init_weight(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, hidden_states):
        global_layer_activation = hidden_states
        local_layer_outputs = []
        for i, (local_layer, global_layer) in enumerate(zip(self.local_layers, self.global_layers)):
            global_layer_activation = global_layer(global_layer_activation)
            local_layer_outputs.append(local_layer(global_layer_activation))
            if i < len(self.global_layers) - 1:
                global_layer_activation = torch.cat((global_layer_activation, hidden_states), 1)

        global_layer_output = self.linear(global_layer_activation)
        local_layer_output = torch.cat(local_layer_outputs, 1)
        combined_output = 0.5 * global_layer_output + 0.5 * local_layer_output
        return global_layer_output, local_layer_output, combined_output


# ckpts3: HMCN3 + Head
class HMCNClassificationHead3(nn.Module):
    def __init__(self, config, class_config):
        super().__init__()
        self.hierarchical_depth = [0, 4096, 4096]
        self.global2local = [0, 4096, 4096]
        self.hierarchical_class = [16, 147]

        self.local_layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        self.q_layers = nn.ModuleList()
        self.k_layers = nn.ModuleList()
        self.v_layers = nn.ModuleList()
        for i in range(1, len(self.hierarchical_depth)):

            global_layer = nn.Sequential(
                nn.Linear(config.hidden_size * i, self.hierarchical_depth[i]),
                nn.ReLU(),
                nn.BatchNorm1d(self.hierarchical_depth[i]),
                nn.Dropout(p=0.1),
                ResidualBlock(self.hierarchical_depth[i],self.hierarchical_depth[i]),
                ResidualBlock(self.hierarchical_depth[i],self.hierarchical_depth[i]),
                nn.Linear(self.hierarchical_depth[i], self.hierarchical_depth[i]),
                
            )
            self.global_layers.append(global_layer)

            local_layer = nn.Sequential(
                nn.Linear(self.hierarchical_depth[i], self.global2local[i]), 
                nn.ReLU(),
                nn.BatchNorm1d(self.global2local[i]),
                nn.Dropout(p=0.1),
                ResidualBlock(self.global2local[i],self.global2local[i]),
                ResidualBlock(self.global2local[i],self.global2local[i]),
                nn.Linear(self.global2local[i], self.hierarchical_class[i-1])
            )
            self.local_layers.append(local_layer)

            if i < len(self.hierarchical_depth) - 1:
                self.q_layers.append(nn.Sequential(
                    nn.Linear(config.hidden_size * i, config.hidden_size), 
                    nn.ReLU(),
                    nn.BatchNorm1d(config.hidden_size),
                ))
                self.k_layers.append(nn.Sequential(
                    nn.Linear(self.hierarchical_class[i-1], config.hidden_size), 
                    nn.ReLU(),
                    nn.BatchNorm1d(config.hidden_size),
                ))
                self.v_layers.append(nn.Sequential(
                    nn.Linear(self.hierarchical_class[i-1], config.hidden_size), 
                    nn.ReLU(),
                    nn.BatchNorm1d(config.hidden_size),
                ))

        self.global_layers.apply(self._init_weight)
        self.local_layers.apply(self._init_weight)
        self.q_layers.apply(self._init_weight)
        self.k_layers.apply(self._init_weight)
        self.v_layers.apply(self._init_weight)

    def _init_weight(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, hidden_states):
        global_layer_activation = hidden_states
        local_layer_outputs = []
        for i, (local_layer, global_layer) in enumerate(zip(self.local_layers, self.global_layers)):
            global_logits = global_layer(global_layer_activation)
            local_logits = local_layer(global_logits)
            if i < len(self.global_layers) - 1:
                q_logits = self.q_layers[i](global_layer_activation)
                k_logits = self.k_layers[i](local_logits)
                v_logits = self.v_layers[i](local_logits)
                atten = torch.softmax(torch.matmul(q_logits, k_logits.transpose(-2, -1)), dim=-1)
                global_layer_activation = torch.matmul(atten, v_logits)
                global_layer_activation = torch.cat((global_layer_activation, hidden_states), 1)
            local_layer_outputs.append(local_logits)

        local_layer_output = torch.cat(local_layer_outputs, 1)
        return local_layer_output, local_layer_output, local_layer_output


class HMCNClassificationHead3_GLU(nn.Module):
    def __init__(self, config, class_config):
        super().__init__()
        self.hierarchical_depth = [0, 2048, 2048]
        self.global2local = [0, 2048, 2048]
        self.hierarchical_class = [48, 582]

        class GEGLU(nn.Module):
            def forward(self, x):
                x, gate = x.chunk(2, dim=-1)
                return x * F.gelu(gate)

        class GLUResidualBlock(nn.Module):
            def __init__(self, in_dim):
                super().__init__()
                self.glu_layer = nn.Sequential(
                    nn.Linear(in_dim, 2*in_dim),
                    GEGLU(), 
                    nn.BatchNorm1d(in_dim),
                    nn.Linear(in_dim, in_dim)
                )
                self.norm = nn.LayerNorm(in_dim)
            def forward(self, x):
                residual = x
                x = self.glu_layer(x)
                x = self.norm(x + residual)
                return x

        self.local_layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        self.q_layers = nn.ModuleList()
        self.k_layers = nn.ModuleList()
        self.v_layers = nn.ModuleList()

        for i in range(1, len(self.hierarchical_depth)):
            global_layer = nn.Sequential(
                nn.Linear(config.hidden_size * i, 2*self.hierarchical_depth[i]),
                GEGLU(),  # 分割维度并门控
                nn.BatchNorm1d(self.hierarchical_depth[i]),
                nn.Dropout(p=0.1),
                GLUResidualBlock(self.hierarchical_depth[i]),  # 使用 GLU 残差块
                nn.Linear(self.hierarchical_depth[i], self.hierarchical_depth[i]),
            )
            self.global_layers.append(global_layer)

            local_layer = nn.Sequential(
                nn.Linear(self.hierarchical_depth[i], 2*self.global2local[i]),
                GEGLU(),
                nn.BatchNorm1d(self.global2local[i]),
                nn.Dropout(p=0.1),
                GLUResidualBlock(self.global2local[i]),  # 替换为 GLU 残差块
                nn.Linear(self.global2local[i], self.hierarchical_class[i-1])
            )
            self.local_layers.append(local_layer)

            if i < len(self.hierarchical_depth) - 1:
                self.q_layers.append(nn.Sequential(
                    nn.Linear(config.hidden_size * i, 2*config.hidden_size),
                    GEGLU(),
                    nn.BatchNorm1d(config.hidden_size),
                ))
                self.k_layers.append(nn.Sequential(
                    nn.Linear(self.hierarchical_class[i-1], 2*config.hidden_size),
                    GEGLU(),
                    nn.BatchNorm1d(config.hidden_size),
                ))
                self.v_layers.append(nn.Sequential(
                    nn.Linear(self.hierarchical_class[i-1], 2*config.hidden_size),
                    GEGLU(),
                    nn.BatchNorm1d(config.hidden_size),
                ))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if 'glu' in m._get_name().lower():
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')  # 适配 GLU
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, hidden_states):
        global_layer_activation = hidden_states
        local_layer_outputs = []
        for i, (local_layer, global_layer) in enumerate(zip(self.local_layers, self.global_layers)):
            global_logits = global_layer(global_layer_activation)
            local_logits = local_layer(global_logits)
            if i < len(self.global_layers) - 1:
                q_logits = self.q_layers[i](global_layer_activation)
                k_logits = self.k_layers[i](local_logits)
                v_logits = self.v_layers[i](local_logits)
                q_logits = F.normalize(q_logits, p=2, dim=-1)
                k_logits = F.normalize(k_logits, p=2, dim=-1)

                atten = torch.softmax(torch.matmul(q_logits, k_logits.transpose(-2, -1)) / (q_logits.size(-1)**0.5), dim=-1)
                global_layer_activation = torch.matmul(atten, v_logits)
                global_layer_activation = torch.cat((global_layer_activation, hidden_states), 1)
            local_layer_outputs.append(local_logits)
        local_layer_output = torch.cat(local_layer_outputs, 1)
        return local_layer_output, local_layer_output, local_layer_output

class ResidualBlock(nn.Module):
    
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out + identity  # 残差连接
        return out


class ClassificationHead2_GLU(nn.Module):
    """GLU-enhanced classification head with hierarchical gating."""
    
    def __init__(self, config, class_config, cluster_nodes_relations, main_numbers, sub_numbers, node_alpha):
        super().__init__()
        self.cluster_nodes_relations = cluster_nodes_relations
        self.main_numbers = main_numbers
        self.sub_numbers = sub_numbers
        self.layer_num = class_config.layer_nums

        class GEGLU(nn.Module):
            def forward(self, x):
                x, gate = x.chunk(2, dim=-1)
                return x * F.gelu(gate)

        class GLUResidualBlock(nn.Module):
            def __init__(self, in_dim):
                super().__init__()
                self.glu_block = nn.Sequential(
                    nn.Linear(in_dim, 2*in_dim),
                    GEGLU(),
                    nn.BatchNorm1d(in_dim),
                    nn.Linear(in_dim, in_dim)
                )
            def forward(self, x):
                return x + self.glu_block(x)

        self.layer_score = nn.Sequential(
            nn.Linear(class_config.layer_nums, config.hidden_size*2),
            GEGLU(),
            nn.BatchNorm1d(config.hidden_size),
            GLUResidualBlock(config.hidden_size),
            GLUResidualBlock(config.hidden_size),
            nn.Linear(config.hidden_size, class_config.node_nums),
        )

        self.gate_network = nn.Sequential(
            nn.Linear(config.hidden_size + class_config.layer_nums, config.hidden_size * 2),
            GEGLU(),
            nn.BatchNorm1d(config.hidden_size),
            nn.Linear(config.hidden_size, class_config.node_nums),
            nn.Sigmoid()  # 保持输出在 [0,1] 范围
        )

        self.hidden_score = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            GEGLU(),
            nn.BatchNorm1d(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            GEGLU(),
            nn.BatchNorm1d(config.hidden_size),
            GLUResidualBlock(config.hidden_size),
            GLUResidualBlock(config.hidden_size),
            nn.Linear(config.hidden_size, class_config.node_nums),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if 'glu' in m._get_name().lower():
                    nn.init.normal_(m.weight, mean=0, std=0.02)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def forward(self, layer_logits, hidden_states):
        node_outputs = self.hidden_score(hidden_states)
        gate_input = torch.cat([hidden_states, layer_logits], dim=1)
        gate_weights = self.gate_network(gate_input)
        
        layer_outputs = self.layer_score(layer_logits)
        
        output = gate_weights * layer_outputs + (1 - gate_weights) * self.out_proj(node_outputs)
        return output


class ClassificationHead2(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, class_config, cluster_nodes_relations, main_numbers, sub_numbers, node_alpha):
        super().__init__()
        self.cluster_nodes_relations = cluster_nodes_relations
        self.main_numbers = main_numbers
        self.sub_numbers = sub_numbers
        self.layer_num = class_config.layer_nums
        self.layer_score = nn.Sequential(
            nn.Linear(class_config.layer_nums, config.hidden_size * 2),
            nn.BatchNorm1d(config.hidden_size * 2),
            nn.ReLU(),
            ResidualBlock(config.hidden_size * 2, config.hidden_size * 2),
            ResidualBlock(config.hidden_size * 2, config.hidden_size * 2),
            nn.Linear(config.hidden_size * 2, class_config.node_nums),
        )
        self.gate_network = nn.Sequential(
            nn.Linear(config.hidden_size + class_config.layer_nums, config.hidden_size * 4),  # 拼接共享特征和粗分类输出
            nn.BatchNorm1d(config.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, class_config.node_nums),
            nn.Sigmoid()
        )

        self.hidden_score = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.BatchNorm1d(config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
        )
        self.batch_norm1 = nn.BatchNorm1d(class_config.node_nums)
        # self.dropout = nn.Dropout(class_config.dropout_rate)
        self.out_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.BatchNorm1d(config.hidden_size * 4),
            nn.ReLU(),
            ResidualBlock(config.hidden_size * 4, config.hidden_size * 4),
            ResidualBlock(config.hidden_size * 4, config.hidden_size * 4),
            nn.Linear(config.hidden_size * 4, class_config.node_nums),
        )

    def forward(self, layer_logits, hidden_states):
        
        node_outputs = self.hidden_score(hidden_states)
        gate_weights = self.gate_network(torch.cat([hidden_states, layer_logits], dim=1))   # 
        layer_outputs = self.layer_score(layer_logits)
        output = gate_weights * layer_outputs + (1 - gate_weights) * self.out_proj(node_outputs)
        return output


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, class_config, cluster_nodes_relations, main_numbers, sub_numbers, node_alpha):
        super().__init__()
        self.cluster_nodes_relations = cluster_nodes_relations
        self.main_numbers = main_numbers
        self.sub_numbers = sub_numbers
        self.layer_num = class_config.layer_nums
        self.layer_score = nn.Sequential(
            nn.Linear(class_config.layer_nums, config.hidden_size * 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size * 2),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
        )
        self.gate_network = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 4),  # 拼接共享特征和粗分类输出
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size * 2),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Sigmoid()
        )

        self.hidden_score = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size * 2),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
        )
        self.batch_norm1 = nn.BatchNorm1d(class_config.node_nums)
        self.dropout = nn.Dropout(class_config.dropout_rate)
        self.out_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size * 4),
            ResidualBlock(config.hidden_size * 4, config.hidden_size * 4),
            ResidualBlock(config.hidden_size * 4, config.hidden_size * 4),
            nn.Linear(config.hidden_size * 4, class_config.node_nums),
        )

    def forward(self, layer_logits, hidden_states):
        layer_logits = self.layer_score(layer_logits)
        hidden_states = self.hidden_score(hidden_states)
        gate_weights = self.gate_network(torch.cat([hidden_states, layer_logits], dim=1))   # 
        hidden_states = gate_weights * hidden_states + (1 - gate_weights) * layer_logits
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

@dataclass
class MySequenceClassifierOutput(ModelOutput):
    layer_loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


def LayerLoss(loss_fn, global_logits, local_logits, layers, cluster_relations, penalty, is_multiss=False, use_hierar=False):
    loss1 = loss_fn(global_logits,
                    layers,
                    use_hierar, # use_hierar
                    is_multiss, # is_multiss
                    penalty, # 惩罚因子
                    global_logits,
                    cluster_relations)
    loss2 = loss_fn(local_logits,
                    layers,
                    use_hierar, # use_hierar
                    is_multiss, # is_multiss
                    penalty, # 惩罚因子
                    local_logits,
                    cluster_relations)

    return loss1[0] + loss2[0], loss2[1] + loss2[1]

def NodeLoss(loss_fn, node_logits, nodes, layer_loss ,bottom_loss, cluster_nodes_relations, is_multi=False, use_hierar=False, hierar_relation=None):
    node_loss = loss_fn(node_logits,
                        nodes,
                        use_hierar,
                        is_multi,
                        1e-2, # 惩罚因子
                        layer_loss,
                        bottom_loss,
                        cluster_nodes_relations,
                        hierar_relation,
                        )
    return node_loss



class Myconfig():
    def __init__(self) -> None:
        self.hidden_size = 4096


class T5EncoderCLSModel2(nn.Module):

    def __init__(self, config, class_config, cluster_relations, hierar_relations, cluster_nodes_relations, main_numbers, sub_numbers, layer_alpha, node_alpha):
        super().__init__()
        self.config = config
        self.use_mlp = False

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.hierar_relations = hierar_relations
        self.cluster_relations = cluster_relations
        self.cluster_nodes_relations = cluster_nodes_relations

        self.penalty = 1e-4

        self.layer_loss_fn = ClassificationLoss(label_size=class_config.layer_nums,\
                                                loss_type=LossType.BCE_WITH_LOGITS,\
                                                layer_alpha=layer_alpha,
                                                node_alpha=node_alpha)
        self.nodes_loss_fn = ClassificationLoss(label_size=class_config.layer_nums + class_config.node_nums, \
                                                loss_type=LossType.BCE_WITH_LOGITS2, \
                                                layer_alpha=layer_alpha,
                                                node_alpha=node_alpha)

        if self.use_mlp:
            self.mlp = MLP(config,class_config)
        else:
            self.layer_classifier = HMCNClassificationHead3_GLU(config, class_config)
            self.node_classifier = ClassificationHead2_GLU(config, class_config, cluster_nodes_relations, main_numbers, sub_numbers, node_alpha)
        # Model parallel
        self.model_parallel = True
        self.device_map = None

    def forward(
        self,
        emb=None,
        layers=None,
        nodes=None
    ):

        hidden_states = emb

        return MySequenceClassifierOutput(
                                    hidden_states=hidden_states,
                                    )

    
    def mean_pooling(self, hidden_states, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask
    
    def max_pooling(self, hidden_states, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        hidden_states[input_mask_expanded == 0] = -1e9
        max_embeddings, _ = torch.max(hidden_states, dim=1)
        return max_embeddings
    

    def save_pretrained(self, save_directory, state_dict, save_func, is_main_process=True):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        
        if is_main_process:
            save_func(state_dict, model_path)
            
            print(f"Model and configuration saved to {save_directory}")


class MLP(nn.Module):
    
    # 

    def __init__(self, config, class_config):
        super().__init__()
        self.layer_num = class_config.layer_nums

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048,class_config.node_nums)
        )

    def forward(self, hidden_states):
        # Hidden 分支特征提取
        output = self.mlp(hidden_states)
        return output