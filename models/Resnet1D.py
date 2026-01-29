'''
ResNet for 1D data.
Adapted from Resnet.py
'''

from collections import OrderedDict
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
from loss.moe_loss import *
from loss.physics_contrast_loss import ICLMoELoss, PhysMoELossWrapper, PhysMoELossWrapperScheme2, PhysMoELossWrapperScheme3
from types import SimpleNamespace
import os
import numpy as _np
from metrics.metrics import accuracy
from utils.utils import core_module
# 移除未使用的HNM模块
# from loss.HNM import HardNegativeMining_Proto as HardNegativeMining
from .CrystalFusionNet import ResTcn, ResBlock1D
from .ViT_Frontend import ViT_Frontend, ViT_Frontend_MeanPool
from .ResNet_Transformer_Frontend import ResNet_Transformer_Frontend


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                # For 1D: x is (N, C, L). x[:, :, ::2] subsamples L.
                # Padding (0, 0, planes // 4, planes // 4) pads C.
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2], (0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class StridedConv(nn.Module):
    """
    downsampling conv layer
    """

    def __init__(self, in_planes, planes, use_relu=False) -> None:
        super(StridedConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_planes, out_channels=planes,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(planes)
        )
        self.use_relu = use_relu
        if use_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)

        if self.use_relu:
            out = self.relu(out)

        return out


class SkipPooling(nn.Module):
    """
    shallow features alignment wrt. depth
    """

    def __init__(self, input_dim=None, depth=None) -> None:
        super(SkipPooling, self).__init__()
        self.convs = nn.Sequential(
            OrderedDict([(f'StridedConv{k}', StridedConv(in_planes=input_dim * (2 ** k), planes=input_dim * (2 ** (k + 1)), use_relu=(k != 1))) for
                         k in range(depth)]))

    def forward(self, x):
        out = self.convs(x)
        return out


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

@core_module
class ResNet1D_MoE(nn.Module):

    def __init__(self, args)->None:
        """
        args:
        args.model.num_experts: number of experts
        args.model.sg_to_sys_map: optional mapping from space-group (0..229) to crystal system (0..6)
        rule_bias：暂时不使用
        tail_expert_ids: 尾部专家对应的id([0...num_experts-1])
        args.model.rule_matrix: precomputed rule similarity matrix
        """
        super(ResNet1D_MoE, self).__init__()
        self.args = args
        #block = BasicBlock
        #num_blocks =  [5, 5, 5]
        num_experts = args.model.num_experts if hasattr(args.model,'num_experts') else 8
        num_classes = 230 
        use_norm=True
        self.s = 30
        self.device = args.train.device
        self.label_dis = args.label_dis 
        # optional mapping from space-group (0..229) to crystal system (0..6)
        # expect args.model.sg_to_sys_map as list or tensor of length num_classes
        if hasattr(args.model, 'sg_to_sys_map'):
            self.sg_to_sys_map = torch.tensor(args.model.sg_to_sys_map, dtype=torch.long)
        else:
            self.sg_to_sys_map = None
        
        # 加载空间群分类信息（Head/Medium/Tail）
        sg_count_path = getattr(args.model, 'sg_count_path', None)
        if sg_count_path is None:
            # 尝试从args中获取
            if hasattr(args, 'rule_matrix_path'):
                # 假设sg_count.csv在rule_matrix_path的同一目录
                import os
                rule_dir = os.path.dirname(args.rule_matrix_path)
                sg_count_path = os.path.join(rule_dir, 'sg_count.csv')
            else:
                sg_count_path = 'sg_count.csv'
        
        from utils.sg_classifier import load_sg_classification
        _, head_mask, medium_mask, tail_mask, extreme_tail_mask = load_sg_classification(
            sg_count_path, num_classes=num_classes
        )
        # 注册为buffer，确保可以移动到GPU
        self.register_buffer('head_mask', head_mask)
        self.register_buffer('medium_mask', medium_mask)
        self.register_buffer('tail_mask', tail_mask)
        self.register_buffer('extreme_tail_mask', extreme_tail_mask)

        # Rule-guided gating bias: shape (7, num_experts) if provided
        if hasattr(args.model, 'rule_bias'):
            self.register_buffer('rule_bias', torch.tensor(args.model.rule_bias, dtype=torch.float))
        else:
            # use local num_experts variable (self.num_experts not set yet)
            self.register_buffer('rule_bias', torch.zeros(7, num_experts))

        self.num_experts = num_experts
        
        # 选择前端网络类型：resnet / vit / resnet_transformer
        frontend_type = getattr(args.model, 'frontend_type', 'resnet').lower()
        
        if frontend_type == 'vit':
            # 使用ViT作为前端
            vit_config = {
                'seq_len': 8500,
                'patch_size': getattr(args.model, 'vit_patch_size', 50),
                'embed_dim': getattr(args.model, 'vit_embed_dim', 256),
                'depth': getattr(args.model, 'vit_depth', 6),
                'num_heads': getattr(args.model, 'vit_num_heads', 8),
                'mlp_ratio': getattr(args.model, 'vit_mlp_ratio', 4),
                'dropout': getattr(args.model, 'p_dropout', 0.1),
                'output_dim': 1024
            }
            # 选择使用CLS token还是mean pooling
            use_cls_token = getattr(args.model, 'vit_use_cls_token', True)
            if use_cls_token:
                self.front_tcn = ViT_Frontend(**vit_config)
            else:
                self.front_tcn = ViT_Frontend_MeanPool(**vit_config)
            print(f"使用ViT前端: patch_size={vit_config['patch_size']}, embed_dim={vit_config['embed_dim']}, "
                  f"depth={vit_config['depth']}, num_heads={vit_config['num_heads']}, "
                  f"use_cls_token={use_cls_token}")
        elif frontend_type == 'resnet_transformer':
            # ResNet 将 8500 缩到约 100，再用 4 层 Transformer，输出 1024
            rt_config = {
                'resnet_out_len': getattr(args.model, 'rt_resnet_out_len', 132),
                'resnet_out_ch': 256,
                'embed_dim': getattr(args.model, 'rt_embed_dim', 256),
                'num_heads': getattr(args.model, 'rt_num_heads', 8),
                'num_layers': getattr(args.model, 'rt_num_layers', 4),
                'mlp_ratio': getattr(args.model, 'rt_mlp_ratio', 4.0),
                'dropout': getattr(args.model, 'rt_dropout', getattr(args.model, 'p_dropout', 0.1)),
                'output_dim': 1024
            }
            self.front_tcn = ResNet_Transformer_Frontend(**rt_config)
            print(f"使用 ResNet+Transformer 前端: resnet_out_len={rt_config['resnet_out_len']}, "
                  f"embed_dim={rt_config['embed_dim']}, num_layers={rt_config['num_layers']}, "
                  f"num_heads={rt_config['num_heads']}")
        else:
            # 使用 ResTcn 前端（仅 ResNet）
            p_dropout = getattr(args.model, 'p_dropout', 0.1) if hasattr(args.model,'p_dropout') else 0.1
            self.res_tcn = ResTcn(in_c=1, p_dropout=p_dropout)
            # keep TCN up to the last ResBlock1D(1024,1024) and AvgPool1d(2,2) (exclude Flatten and Linear)
            tcn_layers = list(self.res_tcn.TCN.children())[:-2]
            self.front_tcn = nn.Sequential(self.res_tcn.intensity_norm, *tcn_layers)
            print(f"使用ResTcn前端: dropout={p_dropout}")

        # 检查是否使用方案2+对比学习（需要轻量化专家）
        loss_scheme = getattr(args.model, 'loss_scheme', 'full')
        use_contrast_scheme2 = getattr(args.model, 'use_contrast_scheme2', False)
        self.use_lightweight_experts = (loss_scheme == 'scheme2' and use_contrast_scheme2)
        
        # 前端输出 channel = 1024
        # 晶系分类器：Flatten -> Linear(1024,32) -> classifier(32->7)
        # 如果使用轻量化专家，扩宽瓶颈：1024 -> 256
        proj_dim = 256 if self.use_lightweight_experts else 32
        self.sys_proj = nn.Sequential(nn.Flatten(), nn.Linear(1024, proj_dim))
        self.sys_classifier = NormedLinear(proj_dim, 7) if use_norm else nn.Linear(proj_dim, 7, bias=True)

        if num_experts:
            if self.use_lightweight_experts:
                # 轻量化专家模式：共享Layer3，专家只负责最后的Linear映射
                # 1. 共享Layer3（单层，不再使用ModuleList）
                self.layer3_shared = ResBlock1D(1024, 1024)
                # 2. 扩宽瓶颈：1024 -> 256（而不是32）
                # 3. 专家只负责最后的分类映射（256 -> 230），不再负责特征提取
                if use_norm:
                    self.expert_proj = None  # 不再需要per-expert投影
                    self.classifiers = nn.ModuleList([NormedLinear(proj_dim, num_classes) for _ in range(self.num_experts)])
                    self.rt_classifiers = nn.ModuleList([NormedLinear(proj_dim, num_classes) for _ in range(self.num_experts)])
                else:
                    self.expert_proj = None
                    self.classifiers = nn.ModuleList([nn.Linear(proj_dim, num_classes, bias=True) for _ in range(self.num_experts)])
                    self.rt_classifiers = nn.ModuleList([nn.Linear(proj_dim, num_classes, bias=True) for _ in range(self.num_experts)])
            else:
                # 原始模式：每个专家有独立的Layer3
                self.layer3s = nn.ModuleList([ResBlock1D(1024, 1024) for _ in range(num_experts)])
                # per-expert projection from 1024 -> 32 then classifier
                if use_norm:
                    self.expert_proj = nn.ModuleList([nn.Sequential(nn.Flatten(), nn.Linear(1024, 32)) for _ in range(self.num_experts)])
                    self.classifiers = nn.ModuleList([NormedLinear(32, num_classes) for _ in range(self.num_experts)])
                    self.rt_classifiers = nn.ModuleList([NormedLinear(32, num_classes) for _ in range(self.num_experts)])
                else:
                    self.expert_proj = nn.ModuleList([nn.Sequential(nn.Flatten(), nn.Linear(1024, 32)) for _ in range(self.num_experts)])
                    self.classifiers = nn.ModuleList([nn.Linear(32, num_classes, bias=True) for _ in range(self.num_experts)])
                    self.rt_classifiers = nn.ModuleList([nn.Linear(32, num_classes, bias=True) for _ in range(self.num_experts)])
            # gating network: from sys feature (proj_dim-d) to gating logits
            self.gating_net = nn.Linear(proj_dim, self.num_experts)
            tail_ids = getattr(args.model, 'tail_expert_ids', list(range(self.num_experts//2, self.num_experts)))
            self.icl_moe = ICLMoELoss(self.num_experts, tail_ids)
        else:
            # single-path head for non-MoE: apply one ResBlock stack then linear
            self.layer3 = nn.Sequential(ResBlock1D(1024, 1024), nn.AvgPool1d(2, 2))
            self.linear = NormedLinear(1024, num_classes) if use_norm else nn.Linear(1024, num_classes, bias=True)

        self.apply(_weights_init)
        # 移除未使用的组件（feature_diversity_loss 和 HNM）
        # self.expert_hnm = [...]
        # self.diversity_loss = feature_diversity_loss

        # prepare rule matrix
        if hasattr(args.model, 'rule_matrix') and args.model.rule_matrix is not None:
            rule_matrix = torch.tensor(args.model.rule_matrix, dtype=torch.float)
        else:
            rule_path = os.path.join(os.getcwd(), 'rule_matrix.csv')
            if os.path.exists(rule_path):
                rule_matrix = torch.tensor(_np.loadtxt(rule_path, delimiter=','), dtype=torch.float)
            else:
                # fallback to identity (no rule similarity)
                rule_matrix = torch.eye(230)

        tail_ids = getattr(args.model, 'tail_expert_ids', list(range(num_experts//2, num_experts)))
        cfg = SimpleNamespace(num_experts=num_experts, tail_expert_ids=tail_ids, feat_dim=1024)
        
        # 根据loss_scheme参数选择损失包装器
        loss_scheme = getattr(args.model, 'loss_scheme', 'full')  # 默认使用完整版本
        use_contrast_scheme2 = getattr(args.model, 'use_contrast_scheme2', False)  # 方案2是否使用对比损失
        if loss_scheme == 'scheme2':
            # 方案2：移除L_reg，保留L_col（可选），可选择是否包含对比损失
            # 获取可配置的损失权重
            lambda_dol = getattr(args.model, 'lambda_dol', 0.5)
            lambda_col = getattr(args.model, 'lambda_col', 0.3)
            lambda_hier = getattr(args.model, 'lambda_hier', 1.5)
            lambda_scl = getattr(args.model, 'lambda_scl', 0.2)
            use_col = getattr(args.model, 'use_col', True)  # 默认启用col loss
            self.phys_wrapper = PhysMoELossWrapperScheme2(
                cfg, rule_matrix, 
                self.sg_to_sys_map if self.sg_to_sys_map is not None else torch.arange(230),
                use_contrast=use_contrast_scheme2,
                lambda_dol=lambda_dol,
                lambda_col=lambda_col,
                lambda_hier=lambda_hier,
                lambda_scl=lambda_scl,
                use_col=use_col
            )
        elif loss_scheme == 'scheme3':
            # 方案3：移除L_reg、L_col和L_sg
            self.phys_wrapper = PhysMoELossWrapperScheme3(cfg, rule_matrix, self.sg_to_sys_map if self.sg_to_sys_map is not None else torch.arange(230))
        elif loss_scheme == 'nocontrast':
            # 无对比学习版本
            from loss.physics_contrast_loss import PhysMoELossWrapperNoContrast
            self.phys_wrapper = PhysMoELossWrapperNoContrast(cfg, rule_matrix, self.sg_to_sys_map if self.sg_to_sys_map is not None else torch.arange(230))
        else:
            # 默认：完整版本（带对比学习）
            self.phys_wrapper = PhysMoELossWrapper(cfg, rule_matrix, self.sg_to_sys_map if self.sg_to_sys_map is not None else torch.arange(230))

    def get_core_params(self)->dict[str,list[float]]:
        """
        Get the core parameters
        """
        return {"ce_weight": [0.1,1],"distill_weight":[0.1,1],"parallel_weight":[0.1,1]}

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        self.next_in_planes = self.in_planes
        for stride in strides:
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, crt=False):
        # Ensure input is (N, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # front_tcn处理
        # ResTcn输出: (B, 1024, L) 需要pooling
        # ViT输出: (B, 1024) 直接使用
        front_output = self.front_tcn(x)
        
        # 判断输出维度并提取特征
        if front_output.dim() == 2:
            # ViT输出: (B, 1024)
            sys_feature = front_output  # (B, 1024)
            front = front_output.unsqueeze(-1)  # (B, 1024, 1) 为了兼容后续代码
        else:
            # ResTcn输出: (B, 1024, L)
            front = front_output
            sys_feature = F.avg_pool1d(front, front.size()[2]).view(front.size(0), -1)  # (B, 1024)
        sys_proj_feature = self.sys_proj(sys_feature)  # (B, proj_dim)
        sys_out = self.s * self.sys_classifier(sys_proj_feature)  # 晶系级输出 (B, 7)
        
        # 保存进入专家模块之前的特征
        self.front_feature = sys_feature

        if self.num_experts:
            if self.use_lightweight_experts:
                # 轻量化专家模式：共享Layer3，专家只负责分类
                # 对于ViT，front_output已经是(B, 1024)，可以直接使用
                # 对于ResTcn，需要经过Layer3处理
                if front_output.dim() == 2:
                    # ViT输出: 直接使用
                    shared_feature = front_output  # (B, 1024)
                else:
                    # ResTcn输出: 需要Layer3处理
                    shared_out3 = self.layer3_shared(front)  # (B, 1024, L')
                    # Adaptive pool to length 1 then flatten -> (B, 1024)
                    shared_feature = F.adaptive_avg_pool1d(shared_out3, 1).view(shared_out3.size(0), -1)  # (B, 1024)
                # 3. 投影到proj_dim（与sys_proj共享相同的投影空间）
                shared_proj_feature = self.sys_proj(shared_feature)  # (B, proj_dim)
                # 4. 每个专家只负责最后的分类映射（256 -> 230）
                if crt == True:
                    outs = [self.s * self.rt_classifiers[i](shared_proj_feature) for i in range(self.num_experts)]
                else:
                    outs = [self.s * self.classifiers[i](shared_proj_feature) for i in range(self.num_experts)]
                # Store features for loss computation
                self.feature = [shared_proj_feature.detach() for _ in range(self.num_experts)]
            else:
                # 原始模式：每个专家有独立的Layer3
                # 对于ViT，front已经是(B, 1024, 1)，可以直接使用
                # 对于ResTcn，需要经过Layer3处理
                if front_output.dim() == 2:
                    # ViT输出: 直接使用，每个专家共享相同特征
                    exp_outs = [front_output] * self.num_experts  # List of (B, 1024)
                else:
                    # ResTcn输出: 需要Layer3处理
                    out3s = [layer(front) for layer in self.layer3s]
                    # Adaptive pool to length 1 then flatten -> (B, 1024)
                    exp_outs = [F.adaptive_avg_pool1d(output, 1).view(output.size(0), -1) for output in out3s]
                # Store features without breaking gradient flow
                self.feature = [f.detach() for f in exp_outs]
                # project to 32-d per expert then classify
                if crt == True:
                    outs = [self.s * self.rt_classifiers[i](self.expert_proj[i](exp_outs[i])) for i in range(self.num_experts)]
                else:
                    outs = [self.s * self.classifiers[i](self.expert_proj[i](exp_outs[i])) for i in range(self.num_experts)]

            # --- gating: rule-guided using sys_feature (projected to proj_dim-d) ---
            # gating logits from gating_net
            gating_logits = self.gating_net(sys_proj_feature)  # (B, K)
            # incorporate rule bias based on predicted sys distribution
            try:
                sys_probs = F.softmax(sys_out, dim=1)  # (B,7)
                # rule_bias: (7, K) -> produce per-sample bias via sys_probs @ rule_bias
                bias = torch.matmul(sys_probs, self.rule_bias)  # (B, K)
                gating_logits = gating_logits + bias
            except Exception:
                # fallback: no bias
                pass

            gating_weights = F.softmax(gating_logits, dim=1)  # (B, K)

            # stacked expert logits: (B, K, C)
            stacked = torch.stack(outs, dim=1)
            # ensemble logits via gating weights
            ensemble_logits = torch.einsum('bk,bkc->bc', gating_weights, stacked)

            # dynamic logit masking based on predicted crystal system (Level-1)
            masked_ensemble = ensemble_logits.clone()
            if self.sg_to_sys_map is not None:
                # predicted sys per sample (take top-1)
                pred_sys = torch.argmax(sys_out, dim=1)
                # build mask: for each sample, allowed sg indices have sg_to_sys_map == pred_sys
                # sg_to_sys_map is cpu tensor; move to device
                sg_map = self.sg_to_sys_map.to(sys_out.device)
                # expand to [B, 230]
                map_exp = sg_map.unsqueeze(0).expand(masked_ensemble.size(0), -1)
                valid_mask = (map_exp == pred_sys.unsqueeze(1))
                # 使用一个更合理的掩码值，避免数值不稳定
                # 计算有效logits的最小值，然后减去一个较大的值作为掩码
                if valid_mask.any():
                    valid_logits_min = masked_ensemble[valid_mask].min() if valid_mask.any() else torch.tensor(-100.0, device=masked_ensemble.device)
                    mask_value = valid_logits_min - 50.0  # 足够小的值，但不会导致数值溢出
                else:
                    mask_value = -100.0
                masked_ensemble = masked_ensemble.masked_fill(~valid_mask, mask_value)

            # store for external losses/inspection
            self.gating_weights = gating_weights
            self.ensemble_logits = ensemble_logits
            self.masked_ensemble_logits = masked_ensemble
            self.expert_logits_list = outs  # 保存专家logits列表，用于loss计算
        else:
            out3 = self.layer3(front)
            out = F.adaptive_avg_pool1d(out3, 1).view(out3.size(0), -1)
            outs = self.linear(out)

        return outs, sys_out

    
    def train_step(self, data:dict[str,torch.Tensor],rt:bool=False,epoch:int=0)->dict[str,torch.Tensor]:
        x, y = data['intensity'].to(torch.float32).to(self.device), data['label'].to(self.device)
        pred, sys_pred = self(x, crt=rt)

        # 使用进入专家模块之前的特征（而不是专家内的特征）
        feats = getattr(self, 'front_feature', None)
        if feats is None:
            # fallback: 重新计算前端特征
            # 确保输入格式正确
            if x.dim() == 2:
                x = x.unsqueeze(1)
            front_output = self.front_tcn(x)
            # 处理ViT和ResTcn的不同输出格式
            if front_output.dim() == 2:
                # ViT输出: (B, 1024)
                feats = front_output
            else:
                # ResTcn输出: (B, 1024, L)
                feats = F.adaptive_avg_pool1d(front_output, 1).view(front_output.size(0), -1)

        # 获取专家logits列表（如果存在）
        expert_logits_list = getattr(self, 'expert_logits_list', None)
        if expert_logits_list is None and self.num_experts:
            # 如果没有保存，使用ensemble_logits作为fallback（但这不是正确的）
            expert_logits_list = [pred] * self.num_experts
        
        outputs = {
            'features': feats,
            'sys_logits': sys_pred,
            'expert_logits': expert_logits_list if expert_logits_list is not None else pred,
            'gating_weights': getattr(self, 'gating_weights', None)
        }
        targets = {'labels': y}

        loss, loss_dict = self.phys_wrapper(outputs, targets)
        # 保存loss_dict供trainer使用
        self.last_loss_dict = loss_dict
        return {'loss': loss, 'loss_dict': loss_dict}
    
    def eval_step(self, data:dict[str,torch.Tensor],rt:bool=False)->dict[str,torch.Tensor]:
        """
        Evaluate the model for one step
        data: input data
        """
        
        x = data['intensity'].to(torch.float32).to(self.device)
        y = data['label'].to(self.device)
        y_onehot = F.one_hot(y, num_classes=230).float()
        pred, sys_pred = self(x,crt=rt)
        
        # 从230类空间群标签转换为7类晶系标签
        if self.sg_to_sys_map is not None:
            sg_to_sys_map_device = self.sg_to_sys_map.to(self.device)
            label7 = sg_to_sys_map_device[y]
        else:
            # 如果没有映射表，手动构建
            sg_to_sys = 2 * [0] + 13 * [1] + 59 * [2] + 68 * [3] + 25 * [4] + 27 * [5] + 36 * [6]
            sg_to_sys_tensor = torch.tensor(sg_to_sys, dtype=torch.long, device=self.device)
            label7 = sg_to_sys_tensor[y]
        
        # 计算晶系分类器的准确率
        sys_acc = accuracy(sys_pred, label7, topk=(1,))[0]
        # accuracy函数返回的是tensor列表，需要提取标量值
        sys_acc = sys_acc.item() if isinstance(sys_acc, torch.Tensor) else sys_acc
        
        # 计算损失时使用未masked的ensemble_logits，避免真实标签被mask导致损失异常
        # 预测时使用masked_ensemble_logits
        if hasattr(self, 'ensemble_logits') and self.ensemble_logits is not None:
            output_for_loss = self.ensemble_logits
        else:
            output_for_loss = sum(pred)/len(pred)
        
        # 计算损失：使用未masked的logits
        loss = F.cross_entropy(output_for_loss, y, reduction='mean')
        
        # 预测时使用masked logits（如果可用）
        if hasattr(self, 'masked_ensemble_logits') and self.masked_ensemble_logits is not None:
            output = self.masked_ensemble_logits.clone()
        elif hasattr(self, 'ensemble_logits') and self.ensemble_logits is not None:
            output = self.ensemble_logits
        else:
            output = sum(pred)/len(pred)
        
        acc1, acc2, acc5 = accuracy(output, y, topk=(1, 2, 5))
        # accuracy函数返回的是tensor列表，需要提取标量值
        acc1 = acc1.item() if isinstance(acc1, torch.Tensor) else acc1
        acc2 = acc2.item() if isinstance(acc2, torch.Tensor) else acc2
        acc5 = acc5.item() if isinstance(acc5, torch.Tensor) else acc5
        
        # 计算空间群的recall和f1score（使用macro平均）
        import numpy as np
        from sklearn.metrics import recall_score, f1_score
        pred_labels = torch.argmax(output, dim=1).cpu().numpy()
        true_labels = y.cpu().numpy()
        # 使用macro平均（对所有类别等权重）
        recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0) * 100.0
        f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0) * 100.0
        
        # 计算三类分类精度（Head/Medium/Tail）和极端尾部类
        from utils.sg_classifier import calculate_class_accuracy
        pred_labels_tensor = torch.argmax(output, dim=1)
        
        # 获取extreme_tail_mask（如果存在）
        extreme_tail_mask = getattr(self, 'extreme_tail_mask', None)
        
        if extreme_tail_mask is not None:
            head_acc, medium_acc, tail_acc, extreme_tail_acc = calculate_class_accuracy(
                pred_labels_tensor, y, self.head_mask, self.medium_mask, self.tail_mask, extreme_tail_mask
            )
            extreme_tail_acc = extreme_tail_acc.item() if isinstance(extreme_tail_acc, torch.Tensor) else extreme_tail_acc
        else:
            head_acc, medium_acc, tail_acc = calculate_class_accuracy(
                pred_labels_tensor, y, self.head_mask, self.medium_mask, self.tail_mask
            )
            extreme_tail_acc = 0.0
        
        head_acc = head_acc.item() if isinstance(head_acc, torch.Tensor) else head_acc
        medium_acc = medium_acc.item() if isinstance(medium_acc, torch.Tensor) else medium_acc
        tail_acc = tail_acc.item() if isinstance(tail_acc, torch.Tensor) else tail_acc
        
        # 同时返回各类的样本数，用于正确加权平均
        device = pred_labels_tensor.device
        head_mask = self.head_mask.to(device)
        medium_mask = self.medium_mask.to(device)
        tail_mask = self.tail_mask.to(device)
        head_samples = head_mask[y]
        medium_samples = medium_mask[y]
        tail_samples = tail_mask[y]
        head_count = head_samples.sum().item()
        medium_count = medium_samples.sum().item()
        tail_count = tail_samples.sum().item()
        
        # 计算极端尾部类的样本数
        extreme_tail_count = 0
        if extreme_tail_mask is not None:
            extreme_tail_mask_device = extreme_tail_mask.to(device)
            extreme_tail_samples = extreme_tail_mask_device[y]
            extreme_tail_count = extreme_tail_samples.sum().item()
        
        exp_acc = []
        for exp_pred in pred:
            exp_acc.append(accuracy(exp_pred, y, topk=(1, 5))[0].item())
        exp_acc = torch.tensor(exp_acc)
        return {
            "pred": output, "loss": loss, 
            "acc1": acc1, "acc2": acc2, "acc5": acc5, 
            "recall": recall, "f1": f1, 
            "head_acc": head_acc, "medium_acc": medium_acc, "tail_acc": tail_acc, "extreme_tail_acc": extreme_tail_acc,
            "head_count": head_count, "medium_count": medium_count, "tail_count": tail_count, "extreme_tail_count": extreme_tail_count,
            'exp_acc':exp_acc,'pred':pred,'label':y,'feature':self.feature, 
            'sys_pred': sys_pred, 'label7': label7, 'sys_acc': sys_acc
        }
    def on_eval_end(self,):
        # 移除未使用的HNM
        # for hnm in self.expert_hnm:
        #     hnm.apply_epoch_momentum()
        pass
    
    def extract_features(self,x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # forward through front TCN
        front_output = self.front_tcn(x)
        # 处理ViT和ResTcn的不同输出格式
        if front_output.dim() == 2:
            # ViT输出: (B, 1024)
            shared_feature = front_output
            front = front_output.unsqueeze(-1)  # (B, 1024, 1) 为了兼容
        else:
            # ResTcn输出: (B, 1024, L)
            front = front_output
        
        if self.num_experts:
            if self.use_lightweight_experts:
                # 轻量化专家模式：使用共享Layer3
                if front_output.dim() == 2:
                    # ViT输出: 直接使用
                    shared_feature = front_output
                else:
                    # ResTcn输出: 需要Layer3处理
                    shared_out3 = self.layer3_shared(front)
                    shared_feature = F.avg_pool1d(shared_out3, shared_out3.size()[2]).view(shared_out3.size(0), -1)
                # 返回共享特征（所有专家使用相同的特征）
                return [shared_feature for _ in range(self.num_experts)]
            else:
                # 原始模式：每个专家有独立的Layer3
                if front_output.dim() == 2:
                    # ViT输出: 直接使用，每个专家共享相同特征
                    exp_outs = [front_output] * self.num_experts  # List of (B, 1024)
                else:
                    # ResTcn输出: 需要Layer3处理
                    out3s = [layer(front) for layer in self.layer3s]
                    exp_outs = [F.avg_pool1d(output, output.size()[2]).view(output.size(0), -1) for output in out3s]
                return exp_outs
        else:
            out3 = self.layer3(front)
            out = F.avg_pool1d(out3, out3.size()[2]).view(out3.size(0), -1)
            return out
        
    def classify(self,feature,crt=True):
        if self.num_experts:
            if self.use_lightweight_experts:
                # 轻量化专家模式：feature已经是投影后的特征，直接分类
                # feature应该是共享的投影特征，需要为每个专家复制
                if isinstance(feature, list):
                    shared_proj_feature = feature[0] if len(feature) > 0 else feature
                else:
                    shared_proj_feature = feature
                if crt == True:
                    outs = [self.s * self.rt_classifiers[i](shared_proj_feature) for i in range(self.num_experts)]
                else:
                    outs = [self.s * self.classifiers[i](shared_proj_feature) for i in range(self.num_experts)]
            else:
                # 原始模式：需要先投影再分类
                if crt == True:
                    outs = [self.s * self.rt_classifiers[i](self.expert_proj[i](feature[i])) for i in range(self.num_experts)]
                else:
                    outs = [self.s * self.classifiers[i](self.expert_proj[i](feature[i])) for i in range(self.num_experts)]
        else:
            outs = self.linear(feature)
        
        return outs
