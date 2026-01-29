#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 LMDB 提取空间群和 XRD 数据，处理并保存为 MP20 格式
- 提取空间群和 XRD 数据
- 处理 XRD 数据为 8500 维（与 MP20 一致）
- 按空间群分布分层分割训练集和测试集（97.3% 训练，2.7% 测试）
- 保存为与 MP20 相同的 .npy 格式
"""
import numpy as np
import lmdb
import pickle
import os
import sys
from collections import defaultdict, namedtuple
from tqdm import tqdm
import argparse

# 定义 CodeRow（用于 pickle 加载）
CodeRow = namedtuple('CodeRow', ['name', 'spacegroup', 'lattice', 'fraccoord', 'cartcoord', 'xray', 'ligands','envs', 'raw_coord'])

# 确保 CodeRow 在 __main__ 模块中可用
if '__main__' in sys.modules:
    sys.modules['__main__'].CodeRow = CodeRow


def process_xrd_data(xray_data, to_xrd_length=8500, min_angle=5.0, max_angle=90.0, step=0.01):
    """
    处理 XRD 数据：将稀疏的 (角度, 强度) 对映射到 8500 个区间
    
    Args:
        xray_data: numpy array, shape [N, 2], 第一列是角度，第二列是强度
        to_xrd_length: 输出长度，默认 8500
        min_angle: 最小角度，默认 5.0
        max_angle: 最大角度，默认 90.0
        step: 角度步长，默认 0.01
    
    Returns:
        xrd_sign: numpy array, shape [8500,], dtype=np.float32
    """
    # 初始化 8500 个 0 值
    xrd_sign = np.zeros(to_xrd_length, dtype=np.float32)
    
    # 提取非零强度的信号（角度、强度对）
    non_zero_mask = xray_data[:, 1] != 0
    if non_zero_mask.sum() > 0:
        angles = xray_data[non_zero_mask, 0]  # 角度
        intensities = xray_data[non_zero_mask, 1]  # 强度
        
        # 将角度映射到对应的区间索引
        # 确保角度在 [min_angle, max_angle] 范围内
        valid_mask = (angles >= min_angle) & (angles <= max_angle)
        if valid_mask.sum() > 0:
            valid_angles = angles[valid_mask]
            valid_intensities = intensities[valid_mask]
            
            # 计算索引：index = int((angle - min_angle) / step)
            indices = ((valid_angles - min_angle) / step).astype(np.int32)
            # 确保索引在有效范围内 [0, to_xrd_length-1]
            indices = np.clip(indices, 0, to_xrd_length - 1)
            
            # 将强度值放入对应区间
            # 如果有多个信号映射到同一区间，使用最大值
            np.maximum.at(xrd_sign, indices, valid_intensities)
    
    return xrd_sign


def load_lmdb_data(lmdb_path, max_samples=None):
    """
    从 LMDB 加载所有数据
    
    Args:
        lmdb_path: LMDB 数据库路径
        max_samples: 最大样本数（用于测试，None 表示加载所有）
    
    Returns:
        features_list: list of numpy arrays, 每个元素是 (8500,) 的 XRD 数据
        labels_list: list of int, 空间群标签（0-indexed）
        spacegroup_counts: dict, 每个空间群的样本数
    """
    print(f"正在从 LMDB 加载数据: {lmdb_path}")
    
    if not os.path.exists(lmdb_path):
        raise FileNotFoundError(f"LMDB 路径不存在: {lmdb_path}")
    
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    
    # 获取总数据量
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        total_count = sum(1 for _ in cursor)
    
    print(f"LMDB 总样本数: {total_count}")
    if max_samples is not None:
        total_count = min(total_count, max_samples)
        print(f"限制加载样本数: {total_count}")
    
    features_list = []
    labels_list = []
    spacegroup_counts = defaultdict(int)
    failed_count = 0
    
    # 确保 CodeRow 在 __main__ 模块中可用
    if '__main__' in sys.modules:
        if not hasattr(sys.modules['__main__'], 'CodeRow'):
            sys.modules['__main__'].CodeRow = CodeRow
    
    with env.begin(write=False) as txn:
        for i in tqdm(range(total_count), desc="加载数据"):
            key = str(i).encode('utf-8')
            row_bytes = txn.get(key)
            
            if row_bytes is None:
                failed_count += 1
                continue
            
            try:
                row = pickle.loads(row_bytes)
                
                # 处理 XRD 数据
                xray_data = row.xray  # shape: [N, 2]
                xrd_sign = process_xrd_data(xray_data)
                
                # 提取空间群标签（转换为 0-indexed）
                spacegroup = row.spacegroup - 1  # 1-indexed -> 0-indexed
                
                # 验证标签有效性
                if spacegroup < 0 or spacegroup >= 230:
                    failed_count += 1
                    continue
                
                features_list.append(xrd_sign)
                labels_list.append(spacegroup)
                spacegroup_counts[spacegroup] += 1
                
            except Exception as e:
                failed_count += 1
                if failed_count <= 10:  # 只打印前10个错误
                    print(f"警告: 无法解析索引 {i} 的数据: {e}")
                continue
    
    env.close()
    
    print(f"\n成功加载: {len(features_list)} 个样本")
    print(f"失败/跳过: {failed_count} 个样本")
    print(f"空间群分布: {len(spacegroup_counts)} 个不同的空间群")
    
    return features_list, labels_list, spacegroup_counts


def stratified_split(features_list, labels_list, train_ratio=0.90, val_ratio=0.075, test_ratio=0.025, random_seed=42):
    """
    按空间群分布进行分层分割（训练集、验证集、测试集）
    
    Args:
        features_list: list of numpy arrays, XRD 特征
        labels_list: list of int, 空间群标签
        train_ratio: 训练集比例，默认 0.90（与 MP20 一致）
        val_ratio: 验证集比例，默认 0.075（与 MP20 一致）
        test_ratio: 测试集比例，默认 0.025（与 MP20 一致）
        random_seed: 随机种子
    
    Returns:
        train_features, train_labels, val_features, val_labels, test_features, test_labels
    """
    # 验证比例总和
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"警告: 比例总和为 {total_ratio:.6f}，将自动归一化")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    print(f"\n按空间群分布进行分层分割")
    print(f"  训练集比例: {train_ratio:.4f} ({train_ratio*100:.2f}%)")
    print(f"  验证集比例: {val_ratio:.4f} ({val_ratio*100:.2f}%)")
    print(f"  测试集比例: {test_ratio:.4f} ({test_ratio*100:.2f}%)")
    
    # 按空间群分组
    spacegroup_data = defaultdict(list)
    for idx, (feature, label) in enumerate(zip(features_list, labels_list)):
        spacegroup_data[label].append((idx, feature, label))
    
    train_features = []
    train_labels = []
    val_features = []
    val_labels = []
    test_features = []
    test_labels = []
    
    np.random.seed(random_seed)
    
    # 对每个空间群进行分割
    for spacegroup, data_list in tqdm(spacegroup_data.items(), desc="分割数据"):
        # 打乱数据
        indices = np.arange(len(data_list))
        np.random.shuffle(indices)
        
        # 计算分割点
        n_total = len(data_list)
        if n_total == 1:
            # 只有1个样本，放入训练集
            train_idx = 1
            val_idx = 1
        elif n_total == 2:
            # 2个样本：1个训练，1个验证
            train_idx = 1
            val_idx = 2
        else:
            # 计算分割点
            train_idx = max(1, int(n_total * train_ratio))
            val_idx = train_idx + max(1, int(n_total * val_ratio))
            # 确保至少有一个测试样本（如果可能）
            if val_idx >= n_total and n_total > 2:
                val_idx = n_total - 1
                train_idx = max(1, val_idx - 1)
        
        # 分割训练集、验证集和测试集
        train_indices = indices[:train_idx]
        val_indices = indices[train_idx:val_idx]
        test_indices = indices[val_idx:]
        
        for idx in train_indices:
            _, feature, label = data_list[idx]
            train_features.append(feature)
            train_labels.append(label)
        
        for idx in val_indices:
            _, feature, label = data_list[idx]
            val_features.append(feature)
            val_labels.append(label)
        
        for idx in test_indices:
            _, feature, label = data_list[idx]
            test_features.append(feature)
            test_labels.append(label)
    
    # 转换为 numpy array
    train_features = np.array(train_features, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int64)
    val_features = np.array(val_features, dtype=np.float32)
    val_labels = np.array(val_labels, dtype=np.int64)
    test_features = np.array(test_features, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.int64)
    
    # 统计空间群分布
    train_sg_counts = defaultdict(int)
    val_sg_counts = defaultdict(int)
    test_sg_counts = defaultdict(int)
    for label in train_labels:
        train_sg_counts[label] += 1
    for label in val_labels:
        val_sg_counts[label] += 1
    for label in test_labels:
        test_sg_counts[label] += 1
    
    total_samples = len(train_labels) + len(val_labels) + len(test_labels)
    print(f"\n分割结果:")
    print(f"  训练集: {len(train_labels)} 个样本 ({len(train_labels)/total_samples*100:.2f}%)")
    print(f"  验证集: {len(val_labels)} 个样本 ({len(val_labels)/total_samples*100:.2f}%)")
    print(f"  测试集: {len(test_labels)} 个样本 ({len(test_labels)/total_samples*100:.2f}%)")
    print(f"  训练集空间群数: {len(train_sg_counts)}")
    print(f"  验证集空间群数: {len(val_sg_counts)}")
    print(f"  测试集空间群数: {len(test_sg_counts)}")
    
    return train_features, train_labels, val_features, val_labels, test_features, test_labels


def save_mp20_format(features, labels, save_path):
    """
    保存为 MP20 格式的 .npy 文件
    
    Args:
        features: numpy array, shape [N, 8500]
        labels: numpy array, shape [N,]
        save_path: 保存路径
    """
    print(f"\n保存数据到: {save_path}")
    
    # 创建目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 按照 MP20 格式保存
    data_dict = {
        'features': features,
        'labels230': labels
    }
    
    np.save(save_path, data_dict, allow_pickle=True)
    print(f"保存完成: {features.shape[0]} 个样本")


def main():
    parser = argparse.ArgumentParser(description='从 LMDB 提取数据并保存为 MP20 格式')
    parser.add_argument('--lmdb_path', type=str, 
                       default='/opt/data/private/xrd2c_data/crystal_ligand_envs_raw',
                       help='LMDB 数据库路径')
    parser.add_argument('--output_base_dir', type=str,
                       default='/opt/data/private/xrd2c_data',
                       help='输出基础目录（将在此目录下创建 ccdc_sg_train 和 ccdc_sg_test 子目录）')
    parser.add_argument('--train_ratio', type=float, default=0.90,
                       help='训练集比例（默认 0.90，与 MP20 一致）')
    parser.add_argument('--val_ratio', type=float, default=0.075,
                       help='验证集比例（默认 0.075，与 MP20 一致）')
    parser.add_argument('--test_ratio', type=float, default=0.025,
                       help='测试集比例（默认 0.025，与 MP20 一致）')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大样本数（用于测试，None 表示处理所有）')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LMDB 数据提取和处理脚本")
    print("=" * 80)
    print(f"LMDB 路径: {args.lmdb_path}")
    print(f"输出基础目录: {args.output_base_dir}")
    print(f"训练集比例: {args.train_ratio}")
    print(f"验证集比例: {args.val_ratio}")
    print(f"测试集比例: {args.test_ratio}")
    print("=" * 80)
    
    # 创建目录结构（与 MP20 一致）
    train_dir = os.path.join(args.output_base_dir, 'ccdc_sg_train')
    test_dir = os.path.join(args.output_base_dir, 'ccdc_sg_test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 1. 从 LMDB 加载数据
    features_list, labels_list, spacegroup_counts = load_lmdb_data(
        args.lmdb_path, 
        max_samples=args.max_samples
    )
    
    if len(features_list) == 0:
        print("错误: 没有成功加载任何数据")
        return
    
    # 2. 按空间群分布进行分层分割
    train_features, train_labels, val_features, val_labels, test_features, test_labels = stratified_split(
        features_list, 
        labels_list, 
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
    
    # 3. 保存为 MP20 格式（目录结构与 MP20 一致）
    # 训练集和验证集保存在 ccdc_sg_train 目录
    train_path = os.path.join(train_dir, 'train_ccdc_sg.npy')
    val_path = os.path.join(train_dir, 'test_val_ccdc_sg.npy')  # 与 MP20 命名一致
    # 测试集保存在 ccdc_sg_test 目录
    test_path = os.path.join(test_dir, 'test_ccdc_sg.npy')
    
    save_mp20_format(train_features, train_labels, train_path)
    save_mp20_format(val_features, val_labels, val_path)
    save_mp20_format(test_features, test_labels, test_path)
    
    # 4. 打印统计信息
    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)
    print(f"目录结构（与 MP20 一致）:")
    print(f"  {train_dir}/")
    print(f"    - train_ccdc_sg.npy")
    print(f"    - test_val_ccdc_sg.npy")
    print(f"  {test_dir}/")
    print(f"    - test_ccdc_sg.npy")
    print(f"\n训练集文件: {train_path}")
    print(f"  样本数: {len(train_labels)}")
    print(f"  特征维度: {train_features.shape}")
    print(f"  空间群数: {len(np.unique(train_labels))}")
    print(f"\n验证集文件: {val_path}")
    print(f"  样本数: {len(val_labels)}")
    print(f"  特征维度: {val_features.shape}")
    print(f"  空间群数: {len(np.unique(val_labels))}")
    print(f"\n测试集文件: {test_path}")
    print(f"  样本数: {len(test_labels)}")
    print(f"  特征维度: {test_features.shape}")
    print(f"  空间群数: {len(np.unique(test_labels))}")
    print("=" * 80)


if __name__ == '__main__':
    main()
