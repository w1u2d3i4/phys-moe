#!/usr/bin/env python3
"""
统计CCDC数据（train、val、test）中的空间群分布情况
生成与sg_count.csv相同格式的表格
"""

import numpy as np
import os
import argparse
from collections import defaultdict
import csv

def load_sg_names_from_standard_list():
    """
    从标准空间群列表加载空间群名称映射
    标准序号：1-230（对应数据中的0-229索引）
    返回：dict，{index: sg_name}，其中index是0-229
    """
    # 标准空间群列表（序号1-230，对应数据索引0-229）
    space_groups = [
        (1, "P1"), (2, "P-1"), (3, "P2"), (4, "P21"), (5, "C2"),
        (6, "Pm"), (7, "Pc"), (8, "Cm"), (9, "Cc"), (10, "P2-m"),
        (11, "P21-m"), (12, "C2-m"), (13, "P2-c"), (14, "P21-c"), (15, "C2-c"),
        (16, "P222"), (17, "P2221"), (18, "P21212"), (19, "P212121"), (20, "C2221"),
        (21, "C222"), (22, "F222"), (23, "I222"), (24, "I212121"), (25, "Pmm2"),
        (26, "Pmc21"), (27, "Pcc2"), (28, "Pma2"), (29, "Pca21"), (30, "Pnc2"),
        (31, "Pmn21"), (32, "Pba2"), (33, "Pna21"), (34, "Pnn2"), (35, "Cmm2"),
        (36, "Cmc21"), (37, "Ccc2"), (38, "Amm2"), (39, "Abm2"), (40, "Ama2"),
        (41, "Aba2"), (42, "Fmm2"), (43, "Fdd2"), (44, "Imm2"), (45, "Iba2"),
        (46, "Ima2"), (47, "Pmmm"), (48, "Pnnn"), (49, "Pccm"), (50, "Pban"),
        (51, "Pmma"), (52, "Pnna"), (53, "Pmna"), (54, "Pcca"), (55, "Pbam"),
        (56, "Pccn"), (57, "Pbcm"), (58, "Pnnm"), (59, "Pmmn"), (60, "Pbcn"),
        (61, "Pbca"), (62, "Pnma"), (63, "Cmcm"), (64, "Cmca"), (65, "Cmmm"),
        (66, "Cccm"), (67, "Cmme"), (68, "Ccce"), (69, "Fmmm"), (70, "Fddd"),
        (71, "Immm"), (72, "Ibam"), (73, "Ibca"), (74, "Imma"), (75, "P4"),
        (76, "P41"), (77, "P42"), (78, "P43"), (79, "I4"), (80, "I41"),
        (81, "P-4"), (82, "I-4"), (83, "P4-m"), (84, "P42-m"), (85, "P4-n"),
        (86, "P42-n"), (87, "I4-m"), (88, "I41-a"), (89, "P422"), (90, "P4212"),
        (91, "P4122"), (92, "P41212"), (93, "P4222"), (94, "P42212"), (95, "P4322"),
        (96, "P43212"), (97, "I422"), (98, "I4122"), (99, "P4mm"), (100, "P4bm"),
        (101, "P42cm"), (102, "P42nm"), (103, "P4cc"), (104, "P4nc"), (105, "P42mc"),
        (106, "P42bc"), (107, "I4mm"), (108, "I4cm"), (109, "I41md"), (110, "I41cd"),
        (111, "P-42m"), (112, "P-42c"), (113, "P-421m"), (114, "P-421c"), (115, "P-4m2"),
        (116, "P-4c2"), (117, "P-4b2"), (118, "P-4n2"), (119, "I-4m2"), (120, "I-4c2"),
        (121, "I-42m"), (122, "I-42d"), (123, "P4-mmm"), (124, "P4-mcc"), (125, "P4-nbm"),
        (126, "P4-nnc"), (127, "P4-mbm"), (128, "P4-mnc"), (129, "P4-nmm"), (130, "P4-ncc"),
        (131, "P42-mmc"), (132, "P42-mcm"), (133, "P42-nbc"), (134, "P42-nnm"), (135, "P42-mbc"),
        (136, "P42-mnm"), (137, "P42-nmc"), (138, "P42-ncm"), (139, "I4-mmm"), (140, "I4-mcm"),
        (141, "I41-amd"), (142, "I41-acd"), (143, "P3"), (144, "P31"), (145, "P32"),
        (146, "R3"), (147, "P-3"), (148, "R-3"), (149, "P312"), (150, "P321"),
        (151, "P3112"), (152, "P3121"), (153, "P3212"), (154, "P3221"), (155, "R32"),
        (156, "P3m1"), (157, "P31m"), (158, "P3c1"), (159, "P31c"), (160, "R3m"),
        (161, "R3c"), (162, "P-31m"), (163, "P-31c"), (164, "P-3m1"), (165, "P-3c1"),
        (166, "R-3m"), (167, "R-3c"), (168, "P6"), (169, "P61"), (170, "P65"),
        (171, "P62"), (172, "P64"), (173, "P63"), (174, "P-6"), (175, "P6-m"),
        (176, "P63-m"), (177, "P622"), (178, "P6122"), (179, "P6522"), (180, "P6222"),
        (181, "P6422"), (182, "P6322"), (183, "P6mm"), (184, "P6cc"), (185, "P63cm"),
        (186, "P63mc"), (187, "P-6m2"), (188, "P-6c2"), (189, "P-62m"), (190, "P-62c"),
        (191, "P6-mmm"), (192, "P6-mcc"), (193, "P63-mcm"), (194, "P63-mmc"), (195, "P23"),
        (196, "F23"), (197, "I23"), (198, "P213"), (199, "I213"), (200, "Pm-3"),
        (201, "Pn-3"), (202, "Fm-3"), (203, "Fd-3"), (204, "Im-3"), (205, "Pa-3"),
        (206, "Ia-3"), (207, "P432"), (208, "P4232"), (209, "F432"), (210, "F4132"),
        (211, "I432"), (212, "P4332"), (213, "P4132"), (214, "I4132"), (215, "P-43m"),
        (216, "F-43m"), (217, "I-43m"), (218, "P-43n"), (219, "F-43c"), (220, "I-43d"),
        (221, "Pm-3m"), (222, "Pn-3n"), (223, "Pm-3n"), (224, "Pn-3m"), (225, "Fm-3m"),
        (226, "Fm-3c"), (227, "Fd-3m"), (228, "Fd-3c"), (229, "Im-3m"), (230, "Ia-3d"),
    ]
    
    # 转换为索引0-229的映射
    sg_names = {}
    for std_idx, sg_name in space_groups:
        data_idx = std_idx - 1  # 标准序号1-230 -> 数据索引0-229
        sg_names[data_idx] = sg_name
    
    return sg_names

def load_sg_names_from_csv(csv_path):
    """
    从sg_count.csv加载空间群名称映射（备用方法）
    返回：dict，{index: sg_name}
    """
    sg_names = {}
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if len(row) > 0 and row[0].strip():
                    sg_names[idx] = row[0].strip()
    return sg_names

def load_npy_data(file_path):
    """
    加载NPY数据文件
    返回：labels230数组
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = np.load(file_path, allow_pickle=True, encoding='latin1')
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()
    
    labels = data.get('labels230')
    if labels is None:
        raise ValueError(f"Key 'labels230' not found in {file_path}")
    
    return labels

def count_space_groups(labels):
    """
    统计空间群分布
    返回：dict，{sg_index: count}
    """
    counts = defaultdict(int)
    labels = np.array(labels)
    
    for label in labels:
        if 0 <= label < 230:
            counts[int(label)] += 1
    
    return counts

def merge_counts(*count_dicts):
    """
    合并多个计数字典
    """
    merged = defaultdict(int)
    for count_dict in count_dicts:
        for sg_idx, count in count_dict.items():
            merged[sg_idx] += count
    return merged

def generate_sg_count_csv(counts, sg_names, output_path):
    """
    生成sg_count.csv格式文件（兼容utils/sg_classifier.py的读取方式）
    格式：空间群序号（标准序号1-230）, 样本数
    注意：sg_classifier.py按行索引读取，所以必须按数据索引0-229的顺序写入
    """
    # 必须按数据索引0-229的顺序写入（即使某些索引没有数据，也要写入0）
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 按数据索引0-229顺序写入，确保行索引与数据索引对应
        for data_idx in range(230):
            std_idx = data_idx + 1  # 数据索引0-229 -> 标准序号1-230
            count = counts.get(data_idx, 0)  # 如果没有数据，写0
            # 格式：标准序号, 样本数
            writer.writerow([std_idx, count])
    
    print(f"已生成统计文件: {output_path} (共230行，按数据索引0-229顺序)")

def classify_space_groups(counts):
    """
    根据样本数将空间群分为Head、Medium、Tail三类
    返回：dict，{'head': [(idx, count), ...], 'medium': [...], 'tail': [...]}
    """
    head = []  # >1000
    medium = []  # 100-1000
    tail = []  # <100
    
    for idx, count in counts.items():
        if count > 1000:
            head.append((idx, count))
        elif count >= 100:
            medium.append((idx, count))
        else:
            tail.append((idx, count))
    
    # 按样本数降序排序
    head.sort(key=lambda x: x[1], reverse=True)
    medium.sort(key=lambda x: x[1], reverse=True)
    tail.sort(key=lambda x: x[1], reverse=True)
    
    return {'head': head, 'medium': medium, 'tail': tail}

def generate_classification_csv(classifications, sg_names, output_path):
    """
    生成分类CSV文件
    格式：类别, 空间群序号（标准序号1-230）, 样本数
    """
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['类别', '空间群序号', '样本数'])
        
        # 按Head、Medium、Tail顺序写入
        for class_name in ['head', 'medium', 'tail']:
            class_label = class_name.capitalize()
            for idx, count in classifications[class_name]:
                std_idx = idx + 1  # 数据索引0-229 -> 标准序号1-230
                writer.writerow([class_label, std_idx, count])
    
    print(f"已生成分类文件: {output_path}")

def print_statistics(counts, dataset_name):
    """
    打印统计信息
    """
    total_samples = sum(counts.values())
    num_sg = len(counts)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1])
    
    print(f"\n=== {dataset_name} 统计 ===")
    print(f"总样本数: {total_samples}")
    print(f"空间群数量: {num_sg}")
    print(f"最少样本数: {sorted_counts[0][1]} (空间群 {sorted_counts[0][0]})")
    print(f"最多样本数: {sorted_counts[-1][1]} (空间群 {sorted_counts[-1][0]})")
    print(f"平均样本数: {total_samples / num_sg:.2f}")

def main():
    parser = argparse.ArgumentParser(description='统计CCDC数据的空间群分布')
    parser.add_argument('--processed_data_dir', type=str, 
                       default='/opt/data/private/xrd2c_data/ccdc_sg',
                       help='CCDC数据目录（包含ccdc_sg_train和ccdc_sg_test子目录）')
    parser.add_argument('--sg_count_csv', type=str,
                       default='/opt/data/private/ICL/sg_count.csv',
                       help='参考的sg_count.csv文件路径（用于获取空间群名称）')
    parser.add_argument('--output', type=str,
                       default='ccdc_sg_count.csv',
                       help='输出CSV文件路径')
    parser.add_argument('--separate', action='store_true',
                       help='是否分别统计train、val、test并生成独立文件')
    
    args = parser.parse_args()
    
    # 加载空间群名称映射（仅用于显示，不用于CSV输出）
    print("加载标准空间群名称映射（仅用于显示）...")
    sg_names = load_sg_names_from_standard_list()
    print(f"加载了 {len(sg_names)} 个空间群名称（标准序号1-230，对应数据索引0-229）")
    
    # 构建文件路径
    train_dir = os.path.join(args.processed_data_dir, 'ccdc_sg_train')
    test_dir = os.path.join(args.processed_data_dir, 'ccdc_sg_test')
    
    train_file = os.path.join(train_dir, 'train_ccdc_sg.npy')
    val_file = os.path.join(train_dir, 'test_val_ccdc_sg.npy')
    test_file = os.path.join(test_dir, 'test_ccdc_sg.npy')
    
    # 检查文件是否存在
    files_to_check = [
        ('train', train_file),
        ('val', val_file),
        ('test', test_file)
    ]
    
    for name, file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"Warning: {name} file not found: {file_path}")
    
    # 加载并统计各数据集
    train_counts = {}
    val_counts = {}
    test_counts = {}
    
    if os.path.exists(train_file):
        print(f"\n加载训练集: {train_file}")
        train_labels = load_npy_data(train_file)
        train_counts = count_space_groups(train_labels)
        print_statistics(train_counts, "训练集")
    
    if os.path.exists(val_file):
        print(f"\n加载验证集: {val_file}")
        val_labels = load_npy_data(val_file)
        val_counts = count_space_groups(val_labels)
        print_statistics(val_counts, "验证集")
    
    if os.path.exists(test_file):
        print(f"\n加载测试集: {test_file}")
        test_labels = load_npy_data(test_file)
        test_counts = count_space_groups(test_labels)
        print_statistics(test_counts, "测试集")
    
    # 合并统计
    all_counts = merge_counts(train_counts, val_counts, test_counts)
    print_statistics(all_counts, "总计（train+val+test）")
    
    # 生成输出文件
    if args.separate:
        # 分别生成文件
        if train_counts:
            train_output = args.output.replace('.csv', '_train.csv')
            generate_sg_count_csv(train_counts, sg_names, train_output)
        
        if val_counts:
            val_output = args.output.replace('.csv', '_val.csv')
            generate_sg_count_csv(val_counts, sg_names, val_output)
        
        if test_counts:
            test_output = args.output.replace('.csv', '_test.csv')
            generate_sg_count_csv(test_counts, sg_names, test_output)
        
        # 生成总计文件
        all_output = args.output.replace('.csv', '_all.csv')
        generate_sg_count_csv(all_counts, sg_names, all_output)
    else:
        # 只生成总计文件
        generate_sg_count_csv(all_counts, sg_names, args.output)
    
    # 打印一些额外信息
    print("\n=== 分布分析 ===")
    sorted_all = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\n样本数最多的前10个空间群:")
    for i, (sg_idx, count) in enumerate(sorted_all[:10], 1):
        sg_name = sg_names.get(sg_idx, f"SG_{sg_idx}")
        print(f"  {i}. {sg_name} (索引{sg_idx}): {count} 个样本")
    
    print(f"\n样本数最少的前10个空间群:")
    for i, (sg_idx, count) in enumerate(sorted_all[-10:], 1):
        sg_name = sg_names.get(sg_idx, f"SG_{sg_idx}")
        print(f"  {i}. {sg_name} (索引{sg_idx}): {count} 个样本")
    
    # 分类空间群
    classifications = classify_space_groups(all_counts)
    
    # 统计Head/Medium/Tail分布
    head_count = len(classifications['head'])
    medium_count = len(classifications['medium'])
    tail_count = len(classifications['tail'])
    
    print(f"\n=== Head/Medium/Tail 分布 ===")
    print(f"Head类 (>1000样本): {head_count} 个空间群")
    print(f"Medium类 (100-1000样本): {medium_count} 个空间群")
    print(f"Tail类 (<100样本): {tail_count} 个空间群")
    
    # 生成分类CSV文件
    classification_output = args.output.replace('.csv', '_classification.csv')
    if args.separate:
        classification_output = args.output.replace('.csv', '_all_classification.csv')
    
    generate_classification_csv(classifications, sg_names, classification_output)
    
    # 打印分类详情
    print(f"\n=== 分类详情 ===")
    for class_name in ['head', 'medium', 'tail']:
        class_label = class_name.capitalize()
        items = classifications[class_name]
        print(f"\n{class_label}类 ({len(items)} 个空间群):")
        print(f"  样本数范围: {items[-1][1]} - {items[0][1]}")
        print(f"  前5个空间群:")
        for i, (idx, count) in enumerate(items[:5], 1):
            sg_name = sg_names.get(idx, f"SG_{idx}")
            print(f"    {i}. {sg_name} (索引{idx}): {count} 个样本")
    
    print("\n统计完成！")

if __name__ == '__main__':
    main()
