'''为了在缺乏实测数据增强的情况下解决极端长尾分布问题，我们提出了一种物理规则约束的混合损失函数。该函数旨在将晶体学的对称性先验注入到特征空间中，并平衡多专家模型的训练动态。总损失函数定义如下：
L_total = λ_1 * L_hier + λ_2 * L_moe + λ_3 * L_phys + λ_4 * L_reg
其中，λ 为各部分的平衡超参数。以下详述各组件的设计逻辑。
3.1 物理规则相似度矩阵 (Physics Rule Matrix) 在计算损失之前，我们首先定义物理世界的“语义真理”。我们构建了一个 C × C 的规则相似度矩阵 S_rule。
对于任意两个空间群类别 i 和 j，S_ij 的值基于它们的晶系归属、点群包含关系及消光规则的重叠度计算得出（范围 [0, 1]）。这是我们将物理先验转化为可微计算的核心基础 。
3.2 物理规则加权的对比损失 (L_phys: Physics-Weighted Contrastive Loss) 针对无实测数据导致的样本多样性匮乏问题，我们改进了传统的监督对比损失（SupCon）。
我们将“正样本”的定义从“同类样本”扩展为“物理语义相似的样本”。具体而言，L_phys 计算如下：
L_phys = - ∑ log ( exp(sim(z_i, z_p) / τ) / ∑ w_ik * exp(sim(z_i, z_k) / τ) )
•	软正样本对 (z_i, z_p)：只要样本 p 与锚点 i 的规则相似度 S_ip 超过阈值（如 0.8），即视为正样本，无论它们是否属于同一类别。
•	规则加权因子 (w_ik)：我们在分母中引入权重 w_ik = 1 - S_ik。这意味着如果负样本 k 与锚点 i 在物理规则上非常相似，我们降低其排斥权重，
避免破坏特征空间的物理连续性。

3.3 独立与协作专家损失 (L_moe: ICL-MoE Loss) 为了防止长尾分布下的专家坍塌，我们采用了独立协作学习（ICL）策略 ，包含两部分：
L_moe = L_dol + β * L_col
1.	独立优化损失 (L_dol)：强制每个专家独立进行分类预测。
o	对于首部专家 (Head Experts)：使用标准交叉熵损失 (CE Loss)，关注通用特征。
o	对于尾部专家 (Tail Experts)：使用 Focal Loss，形式为 (1 - p_t)^γ * log(p_t)，以强化对稀疏难样本的挖掘。
2.	协作蒸馏损失 (L_col)：计算专家输出分布 Q_k 与模型集成分布 Q_ens 之间的 KL 散度，确保知识在专家间有效传递。
3.4 层级化约束损失 (L_hier: Hierarchical Constraint Loss) 基于晶体学的层级结构，我们设计了级联损失：
L_hier = L_sys + L_sg * M_mask
•	L_sys：针对 7 大晶系的粗粒度分类损失。
•	M_mask (层级掩码)：在计算空间群粒度的损失 L_sg 时，我们利用掩码机制屏蔽掉不属于预测晶系的空间群 Logits。这有效地将 230 类分类任务分解为 7 个相对简单的子任务，
显著降低了尾部类的被干扰概率。
3.5 特征中心正则化 (L_reg: Center Regularization) 由于尾部类样本极少，其特征中心容易发生漂移。我们引入基于规则的原型约束：
L_reg = || μ_tail - ∑ (Norm(S_ij) * μ_head) ||^2
该正则项强制尾部类 i 的特征中心 μ_tail 保持在与其物理规则相似的首部类中心的几何邻域内，从而利用首部类的丰富数据来“锚定”尾部类的特征表示。'''

 

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, rule_threshold=0.8):
        super().__init__()
        self.temperature = temperature
        self.rule_threshold = rule_threshold

    def forward(self, features, labels, rule_matrix):
        """
        features: [batch_size, feature_dim] (normalized)
        labels: [batch_size]
        rule_matrix: [num_classes, num_classes] 预计算的规则相似度矩阵
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 1. 计算特征相似度矩阵 (Cosine Similarity)
        # features 必须是 normalize 过的
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 为了数值稳定性，减去最大值
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        # 2. 构建规则相似度 Mask
        # batch_rule_sim[i, j] 表示样本 i 和样本 j 所属类别的规则相似度
        batch_rule_sim = rule_matrix[labels][:, labels].to(device)
        
        # 3. 定义正样本 (Positives)
        # 条件：要么是同类 (Identity)，要么规则极其相似 (Physics Neighbor)
        is_same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
        is_rule_similar = batch_rule_sim > self.rule_threshold
        # 对角线（自己对自己）不参与计算
        mask_self = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        
        # 最终的正样本 Mask
        mask_pos = (is_same_class | is_rule_similar) & mask_self

        # 4. 计算 Logits
        # 分子：exp(sim_pos)
        # 分母：sum( weight * exp(sim_neg) )
        
        # 规则加权因子：越相似的负样本，权重越小 (1 - S)
        # 这样模型不会强行把“物理上相似”的负样本推得太远
        neg_weights = 1.0 - batch_rule_sim 
        # 正样本的权重设为 1 (不影响分子，分子只看 mask_pos)
        weighted_exp = torch.exp(sim_matrix) * neg_weights
        
        # 分母求和 (包含所有样本，除了自己)
        denominator = weighted_exp.sum(dim=1)
        
        # 计算 Log Probability
        # log( exp(pos) / denominator ) = pos - log(denominator)
        log_prob = sim_matrix - torch.log(denominator + 1e-8).unsqueeze(1)
        
        # 只保留正样本对的 Loss
        # mean over positive pairs per anchor
        loss = -(log_prob * mask_pos.float()).sum(dim=1) / (mask_pos.float().sum(dim=1) + 1e-8)
        
        return loss.mean()
    
class ICLMoELoss(nn.Module):
    def __init__(self, num_experts, tail_expert_ids, gamma=2.0, use_col=True):
        super().__init__()
        self.num_experts = num_experts
        self.tail_expert_ids = set(tail_expert_ids) # 标记哪些是尾部专家
        self.gamma = gamma # Focal Loss 参数
        self.use_col = use_col  # 是否使用协作学习损失

    def focal_loss(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

    def forward(self, expert_logits_list, gating_weights, labels):
        """
        expert_logits_list: List of [batch_size, num_classes] (来自 K 个专家)
        gating_weights: [batch_size, num_experts]
        labels: [batch_size]
        """
        
        # --- 1. DoL: 独立优化损失 ---
        # 对专家损失求平均而不是求和，避免损失值过大导致梯度爆炸
        loss_dol = 0.0
        for k, logits in enumerate(expert_logits_list):
            if k in self.tail_expert_ids:
                # 尾部专家用 Focal Loss
                loss_dol += self.focal_loss(logits, labels)
            else:
                # 首部专家用 CE Loss
                loss_dol += F.cross_entropy(logits, labels)
        # 对专家数量求平均，使损失值与单专家模型相当
        loss_dol = loss_dol / len(expert_logits_list)
        
        # --- 2. CoL: 协作学习损失 (KL Divergence) ---
        # 计算集成输出 (Ensemble Logits) - 总是需要用于其他损失
        # stack: [batch, experts, classes]
        stacked_logits = torch.stack(expert_logits_list, dim=1) 
        # 加权求和得到最终预测
        ensemble_logits = torch.einsum('be,bec->bc', gating_weights, stacked_logits)
        
        if self.use_col:
            ensemble_probs = F.softmax(ensemble_logits, dim=1)
            log_ensemble_probs = F.log_softmax(ensemble_logits, dim=1)
            
            loss_col = 0.0
            for logits in expert_logits_list:
                log_expert_probs = F.log_softmax(logits, dim=1)
                # KL(Ensemble || Expert) 让专家向集体看齐
                # 或者 KL(Expert || Ensemble) 让集体吸取专家精华
                # 这里使用双向 JS 散度或者简单的 KL
                loss_col += F.kl_div(log_expert_probs, ensemble_probs, reduction='batchmean')
            # 对专家数量求平均，保持损失值在合理范围
            loss_col = loss_col / len(expert_logits_list)
        else:
            loss_col = torch.tensor(0.0, device=ensemble_logits.device)
            
        return loss_dol, loss_col, ensemble_logits
    
class HierarchicalLoss(nn.Module):
    def __init__(self, sg_to_sys_map, use_sg=True):
        """
        sg_to_sys_map: Tensor [230], 映射每个空间群ID到晶系ID(0-6)
        use_sg: 是否使用空间群级别的损失
        """
        super().__init__()
        self.register_buffer('sg_to_sys_map', sg_to_sys_map)
        self.use_sg = use_sg

    def forward(self, sys_logits, sg_logits, labels):
        # 1. 晶系分类 Loss (Level 1)
        # 获取真实标签对应的晶系标签
        sys_labels = self.sg_to_sys_map[labels]
        loss_sys = F.cross_entropy(sys_logits, sys_labels)
        
        # 2. 空间群分类 Loss (Level 2 - Masked)
        if self.use_sg:
            # 创建一个 Mask: [batch, 230]
            # 如果某空间群属于当前样本的真实晶系，则为 1，否则为 0
            batch_size = labels.size(0)
            num_sg = sg_logits.size(1)
            
            # 扩展映射表以匹配 batch
            # shape: [1, 230]
            map_expanded = self.sg_to_sys_map.unsqueeze(0).expand(batch_size, -1)
            # shape: [batch, 1]
            target_sys = sys_labels.unsqueeze(1)
            
            # valid_mask[i, j] = True if space_group j belongs to system of sample i
            valid_mask = (map_expanded == target_sys)
            
            # Masked Softmax / Cross Entropy
            # 将不属于该晶系的 logits 设为极大负值，使其 softmax 概率为 0
            masked_sg_logits = sg_logits.clone()
            masked_sg_logits[~valid_mask] = -1e9
            #masked_sg_logits[~valid_mask] = -1e4
            
            loss_sg = F.cross_entropy(masked_sg_logits, labels)
        else:
            loss_sg = torch.tensor(0.0, device=sys_logits.device)
        
        return loss_sys, loss_sg
    
class CenterRegularizationLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, rule_matrix):
        super().__init__()
        self.num_classes = num_classes
        # 注册特征中心作为 Parameter (类似 Center Loss)
        # 使用更小的初始化值，避免初始损失过大
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim) * 0.1)
        self.register_buffer('rule_matrix', rule_matrix)

    def forward(self, features, labels):
        # 确保特征和中心都归一化，避免数值不稳定
        # features 已经在外部归一化，这里对centers也归一化
        normalized_centers = F.normalize(self.centers, dim=1)
        batch_centers = normalized_centers[labels]
        
        # 1. 经典的 Center Loss (拉近样本到中心)
        # 使用余弦距离而不是MSE，因为特征已归一化
        # loss_center = F.mse_loss(features, batch_centers)
        # 对于归一化的特征，使用1 - cosine_similarity更合适
        cosine_sim = (features * batch_centers).sum(dim=1)
        loss_center = (1 - cosine_sim).mean()
        
        # 2. 基于规则的原型约束 (Prototype Regularization)
        # 目的：让尾部类的中心不要瞎跑，要向规则相似的首部类看齐
        
        # 找出每个类最相似的 Top-K 个类 (作为物理原型)
        # sim_weights: [230, 230]
        sim_weights = self.rule_matrix.clone()
        sim_weights.fill_diagonal_(0) # 不看自己
        
        # 计算基于规则的“理论中心” (Expected Center)
        # weighted average of other centers based on rule similarity
        # [230, feat_dim] = [230, 230] @ [230, feat_dim]
        # 归一化权重
        weight_sum = sim_weights.sum(dim=1, keepdim=True) + 1e-8
        norm_weights = sim_weights / weight_sum
        
        expected_centers = torch.matmul(norm_weights, normalized_centers)
        
        # 约束：实际中心 应该接近 理论中心
        # 这种约束对于尾部类特别重要，因为它们没有足够的梯度来更新自己
        # 对归一化的中心使用余弦距离
        cosine_sim_reg = (normalized_centers * expected_centers.detach()).sum(dim=1)
        loss_reg = (1 - cosine_sim_reg).mean()
        
        return loss_center + 0.5 * loss_reg
    

class PhysMoELossWrapper(nn.Module):
    def __init__(self, cfg, rule_matrix, sg_to_sys_map):
        super().__init__()
        self.phys_scl = PhysicsContrastiveLoss()
        self.icl_moe = ICLMoELoss(cfg.num_experts, cfg.tail_expert_ids)
        self.hier_loss = HierarchicalLoss(sg_to_sys_map)
        self.center_reg = CenterRegularizationLoss(230, cfg.feat_dim, rule_matrix)
        
        # 权重超参
        self.lambda_scl = 0.1
        self.lambda_hier = 1.0
        self.lambda_reg = 0.001  # 降低中心正则化权重，避免损失过大

    def forward(self, outputs, targets):
        """
        outputs: 字典，包含模型各部分输出
            - 'features': [B, D]
            - 'sys_logits': [B, 7]
            - 'sg_logits': [B, 230]
            - 'expert_logits': List of [B, 230]
            - 'gating_weights': [B, K]
        targets:
            - 'labels': [B] (0-229)
        """
        feats = F.normalize(outputs['features'], dim=1)
        labels = targets['labels']
        
        # 1. 对比学习 Loss
        l_scl = self.phys_scl(feats, labels, self.center_reg.rule_matrix)
        
        # 2. ICL-MoE Loss (返回 DoL, CoL 和 集成Logits)
        l_dol, l_col, ensemble_logits = self.icl_moe(
            outputs['expert_logits'], 
            outputs['gating_weights'], 
            labels
        )
        
        # 3. 层级化 Loss (使用集成后的 Logits 还是 SG Head Logits? 
        # 通常用 Ensemble Logits 作为最终预测)
        l_sys, l_sg = self.hier_loss(outputs['sys_logits'], ensemble_logits, labels)
        
        # 4. 中心正则化
        l_reg = self.center_reg(feats, labels)
        
        # 总 Loss
        total_loss = (
            self.lambda_scl * l_scl +
            l_dol + l_col +           # MoE 核心 Loss
            self.lambda_hier * (l_sys + l_sg) +
            self.lambda_reg * l_reg
        )
        
        return total_loss, {
            "l_scl": l_scl.item(),
            "l_dol": l_dol.item(),
            "l_col": l_col.item(),
            "l_sys": l_sys.item(),
            "l_sg": l_sg.item()
        }


class PhysMoELossWrapperNoContrast(nn.Module):
    """无对比学习版本的损失包装器"""
    def __init__(self, cfg, rule_matrix, sg_to_sys_map):
        super().__init__()
        # 不包含对比学习损失
        self.icl_moe = ICLMoELoss(cfg.num_experts, cfg.tail_expert_ids)
        self.hier_loss = HierarchicalLoss(sg_to_sys_map)
        self.center_reg = CenterRegularizationLoss(230, cfg.feat_dim, rule_matrix)
        
        # 权重超参（去掉对比学习权重）
        self.lambda_hier = 1.0
        self.lambda_reg = 0.001  # 降低中心正则化权重，避免损失过大

    def forward(self, outputs, targets):
        """
        outputs: 字典，包含模型各部分输出
            - 'features': [B, D]
            - 'sys_logits': [B, 7]
            - 'sg_logits': [B, 230]
            - 'expert_logits': List of [B, 230]
            - 'gating_weights': [B, K]
        targets:
            - 'labels': [B] (0-229)
        """
        feats = F.normalize(outputs['features'], dim=1)
        labels = targets['labels']
        
        # 1. ICL-MoE Loss (返回 DoL, CoL 和 集成Logits)
        l_dol, l_col, ensemble_logits = self.icl_moe(
            outputs['expert_logits'], 
            outputs['gating_weights'], 
            labels
        )
        
        # 2. 层级化 Loss (使用集成后的 Logits)
        l_sys, l_sg = self.hier_loss(outputs['sys_logits'], ensemble_logits, labels)
        
        # 3. 中心正则化
        l_reg = self.center_reg(feats, labels)
        
        # 总 Loss（去掉对比学习部分）
        total_loss = (
            l_dol + l_col +           # MoE 核心 Loss
            self.lambda_hier * (l_sys + l_sg) +
            self.lambda_reg * l_reg
        )
        
        return total_loss, {
            "l_dol": l_dol.item(),
            "l_col": l_col.item(),
            "l_sys": l_sys.item(),
            "l_sg": l_sg.item(),
            "l_reg": l_reg.item()
        }


class PhysMoELossWrapperScheme2(nn.Module):
    """方案2：简化版本损失包装器
    移除: L_reg (中心正则化)
    保留: L_dol (独立优化), L_col (协作学习，可选), L_sys (晶系损失), L_sg (空间群损失)
    """
    def __init__(self, cfg, rule_matrix, sg_to_sys_map, use_contrast=False, 
                 lambda_dol=0.5, lambda_col=0.3, lambda_hier=1.5, lambda_scl=0.2,
                 use_col=True):
        super().__init__()
        # 保存rule_matrix用于对比损失
        self.register_buffer('rule_matrix', rule_matrix)
        # 根据use_contrast参数决定是否包含对比学习损失
        self.use_contrast = use_contrast
        if use_contrast:
            self.phys_scl = PhysicsContrastiveLoss()
            self.lambda_scl = lambda_scl
        else:
            self.lambda_scl = 0.0
        # 根据use_col参数决定是否使用协作学习损失
        self.use_col = use_col
        self.icl_moe = ICLMoELoss(cfg.num_experts, cfg.tail_expert_ids, use_col=use_col)
        self.hier_loss = HierarchicalLoss(sg_to_sys_map, use_sg=True)  # 使用L_sg
        
        # 权重超参（可配置）
        self.lambda_dol = lambda_dol  # MoE独立优化损失权重
        # 如果禁用col loss，将权重设为0
        self.lambda_col = lambda_col if use_col else 0.0  # 协作学习损失权重
        self.lambda_hier = lambda_hier  # 层级化损失权重

    def forward(self, outputs, targets):
        """
        outputs: 字典，包含模型各部分输出
            - 'features': [B, D]
            - 'sys_logits': [B, 7]
            - 'sg_logits': [B, 230]
            - 'expert_logits': List of [B, 230]
            - 'gating_weights': [B, K]
        targets:
            - 'labels': [B] (0-229)
        """
        labels = targets['labels']
        
        # 0. 对比学习 Loss (如果启用)
        if self.use_contrast:
            feats = F.normalize(outputs['features'], dim=1)
            # 需要rule_matrix，从center_reg获取或直接传入
            # 这里我们需要rule_matrix，但方案2没有center_reg
            # 所以需要从外部传入或使用一个临时对象
            # 为了简化，我们创建一个临时的CenterRegularizationLoss来获取rule_matrix
            # 或者直接从__init__传入的rule_matrix使用
            # 实际上PhysicsContrastiveLoss需要rule_matrix作为参数
            # 让我们检查一下PhysicsContrastiveLoss的forward签名
            # 从代码看，PhysicsContrastiveLoss.forward(feats, labels, rule_matrix)
            # 我们需要在__init__中保存rule_matrix
            l_scl = self.phys_scl(feats, labels, self.rule_matrix)
        else:
            l_scl = torch.tensor(0.0, device=outputs['features'].device)
        
        # 1. ICL-MoE Loss (返回 DoL 和 CoL)
        l_dol, l_col, ensemble_logits = self.icl_moe(
            outputs['expert_logits'], 
            outputs['gating_weights'], 
            labels
        )
        
        # 2. 层级化 Loss (使用集成后的 Logits)
        l_sys, l_sg = self.hier_loss(outputs['sys_logits'], ensemble_logits, labels)
        
        # 总 Loss（方案2：L_dol + L_col + L_sys + L_sg [+ L_scl如果启用]）
        total_loss = (
            self.lambda_dol * l_dol +                    # MoE 独立优化 Loss（加权）
            self.lambda_col * l_col +                    # 协作学习 Loss（加权，如果启用）
            self.lambda_hier * (l_sys + l_sg)            # 层级化 Loss
        )
        if self.use_contrast:
            total_loss = total_loss + self.lambda_scl * l_scl
        
        return total_loss, {
            "l_scl": l_scl.item() if self.use_contrast else 0.0,
            "l_dol": l_dol.item(),
            "l_col": l_col.item() if self.use_col else 0.0,  # 协作学习损失（如果启用）
            "l_sys": l_sys.item(),
            "l_sg": l_sg.item(),
            "l_reg": 0.0   # 不使用
        }


class PhysMoELossWrapperScheme3(nn.Module):
    """方案3：极简版本损失包装器
    移除: L_reg (中心正则化), L_col (协作学习), L_sg (空间群损失)
    保留: L_dol (独立优化), L_sys (晶系损失)
    """
    def __init__(self, cfg, rule_matrix, sg_to_sys_map):
        super().__init__()
        # 不包含对比学习损失、中心正则化和协作学习
        self.icl_moe = ICLMoELoss(cfg.num_experts, cfg.tail_expert_ids, use_col=False)  # 不使用L_col
        self.hier_loss = HierarchicalLoss(sg_to_sys_map, use_sg=False)  # 不使用L_sg
        
        # 权重超参
        self.lambda_hier = 1.0

    def forward(self, outputs, targets):
        """
        outputs: 字典，包含模型各部分输出
            - 'features': [B, D]
            - 'sys_logits': [B, 7]
            - 'sg_logits': [B, 230]
            - 'expert_logits': List of [B, 230]
            - 'gating_weights': [B, K]
        targets:
            - 'labels': [B] (0-229)
        """
        labels = targets['labels']
        
        # 1. ICL-MoE Loss (只返回 DoL，不包含 CoL)
        l_dol, l_col, ensemble_logits = self.icl_moe(
            outputs['expert_logits'], 
            outputs['gating_weights'], 
            labels
        )
        
        # 2. 层级化 Loss (只使用 L_sys，不使用 L_sg)
        l_sys, l_sg = self.hier_loss(outputs['sys_logits'], ensemble_logits, labels)
        
        # 总 Loss（方案3：L_dol + L_sys）
        total_loss = (
            l_dol +                    # MoE 独立优化 Loss
            self.lambda_hier * l_sys   # 只使用晶系损失
        )
        
        return total_loss, {
            "l_dol": l_dol.item(),
            "l_col": 0.0,  # 不使用
            "l_sys": l_sys.item(),
            "l_sg": 0.0,   # 不使用
            "l_reg": 0.0   # 不使用
        }