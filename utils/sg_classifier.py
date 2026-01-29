"""
空间群分类工具：根据样本数将空间群分为Head、Medium、Tail三类
使用基于CCDC数据集的比例来计算阈值，确保不同数据集使用相同的划分比例
"""
import torch
import csv
import os

# CCDC数据集的基准阈值（用于计算比例）
CCDC_BASE_TOTAL = 120588  # CCDC数据集总样本数
CCDC_HEAD_THRESHOLD = 1000  # Head类阈值
CCDC_MEDIUM_THRESHOLD = 100  # Medium类阈值
CCDC_TAIL_THRESHOLD = 100  # Tail类阈值（<100）
CCDC_EXTREME_TAIL_THRESHOLD = 20  # Extreme Tail类阈值（<20）

def _calculate_thresholds_from_ccdc_ratio(total_samples):
    """
    根据CCDC数据集的比例计算当前数据集的阈值
    
    Args:
        total_samples: 当前数据集的总样本数
    
    Returns:
        head_threshold: Head类阈值
        medium_threshold: Medium类阈值（最小值）
        tail_threshold: Tail类阈值（最大值）
        extreme_tail_threshold: Extreme Tail类阈值（最大值）
    """
    # 计算比例
    head_ratio = CCDC_HEAD_THRESHOLD / CCDC_BASE_TOTAL
    medium_ratio = CCDC_MEDIUM_THRESHOLD / CCDC_BASE_TOTAL
    extreme_tail_ratio = CCDC_EXTREME_TAIL_THRESHOLD / CCDC_BASE_TOTAL
    
    # 应用到当前数据集
    head_threshold = int(total_samples * head_ratio)
    medium_threshold = int(total_samples * medium_ratio)
    extreme_tail_threshold = int(total_samples * extreme_tail_ratio)
    
    # 确保阈值至少为1（避免为0）
    head_threshold = max(1, head_threshold)
    medium_threshold = max(1, medium_threshold)
    extreme_tail_threshold = max(1, extreme_tail_threshold)
    
    return head_threshold, medium_threshold, extreme_tail_threshold

def load_sg_classification(csv_path, num_classes=230):
    """
    从CSV文件加载空间群样本数，并分类
    使用基于CCDC数据集的比例来计算阈值，确保不同数据集使用相同的划分比例
    
    Args:
        csv_path: sg_count.csv文件路径
        num_classes: 空间群类别数（默认230）
    
    Returns:
        sg_counts: [num_classes] 每个空间群的样本数
        head_mask: [num_classes] bool tensor, Head类 (基于比例计算的阈值)
        medium_mask: [num_classes] bool tensor, Medium类 (基于比例计算的阈值)
        tail_mask: [num_classes] bool tensor, Tail类 (基于比例计算的阈值)
        extreme_tail_mask: [num_classes] bool tensor, Extreme Tail类 (基于比例计算的阈值)
    """
    sg_counts = torch.zeros(num_classes, dtype=torch.long)
    head_mask = torch.zeros(num_classes, dtype=torch.bool)
    medium_mask = torch.zeros(num_classes, dtype=torch.bool)
    tail_mask = torch.zeros(num_classes, dtype=torch.bool)
    extreme_tail_mask = torch.zeros(num_classes, dtype=torch.bool)
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, using default classification")
        # 默认分类：假设前50%是Head，中间30%是Medium，后20%是Tail
        head_mask[:115] = True
        medium_mask[115:184] = True
        tail_mask[184:] = True
        return sg_counts, head_mask, medium_mask, tail_mask, extreme_tail_mask
    
    # 首先读取所有样本数，计算总样本数
    total_samples = 0
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx >= num_classes:
                break
            if len(row) >= 2 and row[1].strip():
                try:
                    count = int(row[1].strip())
                    sg_counts[idx] = count
                    total_samples += count
                except ValueError:
                    pass
    
    # 如果总样本数为0，使用默认分类
    if total_samples == 0:
        print(f"Warning: Total samples is 0 in {csv_path}, using default classification")
        head_mask[:115] = True
        medium_mask[115:184] = True
        tail_mask[184:] = True
        return sg_counts, head_mask, medium_mask, tail_mask, extreme_tail_mask
    
    # 根据CCDC数据集的比例计算阈值
    head_threshold, medium_threshold, extreme_tail_threshold = _calculate_thresholds_from_ccdc_ratio(total_samples)
    
    print(f"数据集总样本数: {total_samples}")
    print(f"基于CCDC比例计算的阈值: Head>{head_threshold}, Medium>={medium_threshold}, Tail<{medium_threshold}, Extreme Tail<{extreme_tail_threshold}")
    
    # 重新读取文件进行分类
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx >= num_classes:
                break
            if len(row) >= 2 and row[1].strip():
                try:
                    count = int(row[1].strip())
                    
                    # 分类（使用基于比例计算的阈值）
                    if count > head_threshold:
                        head_mask[idx] = True
                    elif count >= medium_threshold:
                        medium_mask[idx] = True
                    else:
                        tail_mask[idx] = True
                    
                    if count < extreme_tail_threshold:
                        extreme_tail_mask[idx] = True
                except ValueError:
                    pass
    
    return sg_counts, head_mask, medium_mask, tail_mask, extreme_tail_mask


def calculate_class_accuracy(pred_labels, true_labels, head_mask, medium_mask, tail_mask, extreme_tail_mask=None):
    """
    计算三类（Head、Medium、Tail）和极端尾部类的准确率
    
    Args:
        pred_labels: [batch_size] 预测标签
        true_labels: [batch_size] 真实标签
        head_mask: [num_classes] Head类掩码
        medium_mask: [num_classes] Medium类掩码
        tail_mask: [num_classes] Tail类掩码
        extreme_tail_mask: [num_classes] Extreme Tail类掩码（可选，<20样本）
    
    Returns:
        head_acc: Head类准确率（百分比）
        medium_acc: Medium类准确率（百分比）
        tail_acc: Tail类准确率（百分比）
        extreme_tail_acc: Extreme Tail类准确率（百分比，如果extreme_tail_mask提供）
    """
    device = pred_labels.device
    head_mask = head_mask.to(device)
    medium_mask = medium_mask.to(device)
    tail_mask = tail_mask.to(device)
    
    # 获取每类样本的掩码
    head_samples = head_mask[true_labels]
    medium_samples = medium_mask[true_labels]
    tail_samples = tail_mask[true_labels]
    
    # 计算每类的正确预测数
    head_correct = ((pred_labels == true_labels) & head_samples).sum().float()
    medium_correct = ((pred_labels == true_labels) & medium_samples).sum().float()
    tail_correct = ((pred_labels == true_labels) & tail_samples).sum().float()
    
    # 计算每类的样本数
    head_count = head_samples.sum().float()
    medium_count = medium_samples.sum().float()
    tail_count = tail_samples.sum().float()
    
    # 计算准确率（百分比）
    head_acc = (head_correct / head_count * 100.0) if head_count > 0 else torch.tensor(0.0, device=device)
    medium_acc = (medium_correct / medium_count * 100.0) if medium_count > 0 else torch.tensor(0.0, device=device)
    tail_acc = (tail_correct / tail_count * 100.0) if tail_count > 0 else torch.tensor(0.0, device=device)
    
    # 计算极端尾部类准确率（如果提供mask）
    extreme_tail_acc = None
    if extreme_tail_mask is not None:
        extreme_tail_mask = extreme_tail_mask.to(device)
        extreme_tail_samples = extreme_tail_mask[true_labels]
        extreme_tail_correct = ((pred_labels == true_labels) & extreme_tail_samples).sum().float()
        extreme_tail_count = extreme_tail_samples.sum().float()
        extreme_tail_acc = (extreme_tail_correct / extreme_tail_count * 100.0) if extreme_tail_count > 0 else torch.tensor(0.0, device=device)
        return head_acc, medium_acc, tail_acc, extreme_tail_acc
    
    return head_acc, medium_acc, tail_acc
