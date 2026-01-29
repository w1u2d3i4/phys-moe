"""
LMDB数据集类，用于加载xrd2c_v2项目的LMDB数据
兼容ICL训练框架的数据格式
"""
from torch.utils.data import Dataset
import torch
import numpy as np
import lmdb
import pickle
import csv
import os
import sys
from collections import namedtuple

CodeRow = namedtuple('CodeRow', ['name', 'spacegroup', 'lattice', 'fraccoord', 'cartcoord', 'xray', 'ligands','envs', 'raw_coord'])

# 确保CodeRow在__main__模块中可用（用于pickle加载）
if '__main__' in sys.modules:
    sys.modules['__main__'].CodeRow = CodeRow


def xrd_process(xrd, resolution=None, src_len=None, min_angle=5.0, max_angle=90.0, step=0.01, sigma=0.1):
    """
    Convert sparse PXRD inputs into a fixed-grid dense vector using Gaussian broadening.
    
    Args:
        xrd: Input XRD data (numpy array or torch tensor), shape [N, 2]
             - Column 0: 2-theta angles (degrees)
             - Column 1: Intensities
        resolution: (deprecated, kept for compatibility) Not used in new implementation
        src_len: (deprecated, kept for compatibility) Not used in new implementation
        min_angle: Minimum 2-theta angle (degrees), default=5.0
        max_angle: Maximum 2-theta angle (degrees), default=90.0
        step: Grid resolution (degrees), default=0.01
        sigma: Standard deviation for Gaussian broadening (degrees), default=0.1
    
    Returns:
        pos_emb: Position embeddings (grid indices), shape [grid_length, 1]
        sign_emb: Signal embeddings (broadened and normalized intensities), shape [grid_length, 1]
    """
    # Convert to torch tensor if numpy array
    if isinstance(xrd, np.ndarray):
        xrd = torch.from_numpy(xrd).float()
    else:
        xrd = xrd.float()
    
    # Find all signals (peaks with non-zero intensity) within the specified angle range
    signal_mask = (xrd[:, 1] != 0) & (xrd[:, 0] >= min_angle) & (xrd[:, 0] <= max_angle)
    if signal_mask.sum() == 0:
        # No signals found in range, return zero-filled arrays
        grid_length = int((max_angle - min_angle) / step) + 1
        pos_emb = torch.arange(grid_length, dtype=torch.float).unsqueeze(1)
        sign_emb = torch.zeros((grid_length, 1), dtype=torch.float)
        return pos_emb, sign_emb
    
    # Extract peaks (angles and intensities) within range
    peak_angles = xrd[signal_mask, 0]  # [num_peaks]
    peak_intensities = xrd[signal_mask, 1]  # [num_peaks]
    
    # Create fixed grid
    grid_length = int((max_angle - min_angle) / step) + 1
    grid_angles = torch.linspace(min_angle, max_angle, grid_length)  # [grid_length]
    
    # Apply Gaussian broadening using broadcasting
    # Shape: [num_peaks, grid_length]
    # For each peak, compute Gaussian contribution at each grid point
    diff = grid_angles.unsqueeze(0) - peak_angles.unsqueeze(1)  # [num_peaks, grid_length]
    gaussian_weights = torch.exp(-0.5 * (diff / sigma) ** 2)  # [num_peaks, grid_length]
    
    # Weight by peak intensities and sum across all peaks
    # Shape: [grid_length]
    broadened_intensities = (gaussian_weights * peak_intensities.unsqueeze(1)).sum(dim=0)
    
    # Normalize intensities to [0, 1]
    max_intensity = broadened_intensities.max()
    if max_intensity > 0:
        normalized_intensities = broadened_intensities / max_intensity
    else:
        normalized_intensities = broadened_intensities
    
    # Create position embeddings (grid indices)
    pos_emb = torch.arange(grid_length, dtype=torch.float).unsqueeze(1)  # [grid_length, 1]
    sign_emb = normalized_intensities.unsqueeze(1)  # [grid_length, 1]
    
    return pos_emb, sign_emb


class LMDBXrdData(Dataset):
    """
    LMDB数据集类，兼容ICL训练框架
    返回格式与XrdData相同：{'intensity': ..., 'label': ...}
    """
    cls_num = 230
    
    def __init__(self, lmdb_path, train=True, train_ratio=0.8, val_ratio=0.1, sg_count_path=None):
        """
        Args:
            lmdb_path: LMDB数据库路径
            train: 是否为训练集
            train_ratio: 训练集比例
            val_ratio: 验证集比例（剩余为测试集）
            sg_count_path: sg_count.csv文件路径，用于读取类别分布
        """
        self.lmdb_path = lmdb_path
        self.train = train
        self.sg_count_path = sg_count_path
        
        # 打开LMDB数据库
        self.env = lmdb.open(
            lmdb_path,
            max_readers=32,
            readonly=True,  # 只读模式
            lock=False,
            readahead=False,
            meminit=False,
        )
        
        if not self.env:
            raise IOError(f'Cannot open lmdb dataset: {lmdb_path}')
        
        # 获取数据集总长度
        with self.env.begin(write=False) as txn:
            length_bytes = txn.get('length'.encode('utf-8'))
            if length_bytes is None:
                raise ValueError(f"LMDB database at {lmdb_path} does not contain 'length' key")
            self.total_length = int(length_bytes.decode('utf-8'))
        
        # 计算数据集划分
        train_size = int(self.total_length * train_ratio)
        val_size = int(self.total_length * val_ratio)
        test_size = self.total_length - train_size - val_size
        
        if train:
            self.start_idx = 0
            self.end_idx = train_size
            self.length = train_size
        else:
            # 验证集和测试集都使用val模式（可以根据需要进一步划分）
            self.start_idx = train_size
            self.end_idx = train_size + val_size
            self.length = val_size
        
        # 从sg_count.csv读取类别分布
        self.num_classes = 230
        self.cls_num_list = self._load_cls_num_list_from_csv()
        
        # 为了兼容trainer代码，需要提供targets属性
        # 延迟加载targets（只在需要时加载）
        self._targets = None
    
    def __len__(self):
        return self.length
    
    @property
    def targets(self):
        """延迟加载targets属性（兼容trainer代码）"""
        if self._targets is None:
            # 从LMDB加载所有标签
            print(f"Loading targets for {'train' if self.train else 'val'} set (length={self.length})...")
            self._targets = []
            for i in range(self.length):
                actual_index = self.start_idx + i
                try:
                    with self.env.begin(write=False) as txn:
                        key = str(actual_index).encode('utf-8')
                        row_bytes = txn.get(key)
                        if row_bytes is not None:
                            try:
                                # CodeRow已经在模块导入时添加到__main__模块中
                                row = pickle.loads(row_bytes)
                                sg_label = row.spacegroup - 1
                                if 0 <= sg_label < self.num_classes:
                                    self._targets.append(sg_label)
                                else:
                                    self._targets.append(0)  # 默认值
                            except Exception as e:
                                # 如果无法解析，使用默认值
                                self._targets.append(0)  # 默认值
                        else:
                            self._targets.append(0)  # 默认值
                except Exception as e:
                    self._targets.append(0)  # 默认值
            self._targets = np.array(self._targets, dtype=np.int64)
            print(f"Loaded {len(self._targets)} targets")
        return self._targets
    
    def __getitem__(self, index):
        """
        返回格式与XrdData兼容：{'intensity': ..., 'label': ...}
        """
        # 转换为LMDB中的实际索引
        actual_index = self.start_idx + index
        
        with self.env.begin(write=False) as txn:
            key = str(actual_index).encode('utf-8')
            row_bytes = txn.get(key)
            if row_bytes is None:
                raise IndexError(f"Index {actual_index} not found in LMDB")
            # 使用pickle加载数据
            # 确保CodeRow在__main__模块中可用（pickle需要在这里找到它）
            if '__main__' in sys.modules:
                if not hasattr(sys.modules['__main__'], 'CodeRow'):
                    sys.modules['__main__'].CodeRow = CodeRow
            
            try:
                row = pickle.loads(row_bytes)
            except AttributeError as e:
                if "CodeRow" in str(e):
                    # 如果仍然无法找到CodeRow，强制添加到__main__模块
                    if '__main__' in sys.modules:
                        sys.modules['__main__'].CodeRow = CodeRow
                    # 也添加到当前模块
                    if not hasattr(sys.modules[__name__], 'CodeRow'):
                        sys.modules[__name__].CodeRow = CodeRow
                    # 重试加载
                    row = pickle.loads(row_bytes)
                else:
                    raise
        
        # 处理XRD数据：直接映射到8500个区间（与MP20格式一致）
        # 不使用xrd_process，而是直接根据角度将强度映射到对应区间
        TO_XRD_LENGTH = 8500
        min_angle = 5.0
        max_angle = 90.0
        step = 0.01
        
        # 初始化8500个0值
        xrd_sign = np.zeros(TO_XRD_LENGTH, dtype=np.float32)
        
        # 提取xray数据：选择第二列（强度）
        xray_data = row.xray  # shape: [N, 2]
        
        # 提取非零强度的信号（角度、强度对）
        # xray[:, 0] 是角度，xray[:, 1] 是强度
        non_zero_mask = xray_data[:, 1] != 0
        if non_zero_mask.sum() > 0:
            angles = xray_data[non_zero_mask, 0]  # 角度
            intensities = xray_data[non_zero_mask, 1]  # 强度
            
            # 将角度映射到对应的区间索引
            # 索引计算：index = int((angle - min_angle) / step)
            # 确保角度在 [min_angle, max_angle] 范围内
            valid_mask = (angles >= min_angle) & (angles <= max_angle)
            if valid_mask.sum() > 0:
                valid_angles = angles[valid_mask]
                valid_intensities = intensities[valid_mask]
                
                # 计算索引：index = int((angle - 5.0) / 0.01)
                indices = ((valid_angles - min_angle) / step).astype(np.int32)
                # 确保索引在有效范围内 [0, TO_XRD_LENGTH-1]
                indices = np.clip(indices, 0, TO_XRD_LENGTH - 1)
                
                # 将强度值放入对应区间
                # 如果有多个信号映射到同一区间，使用最大值（使用 np.maximum.at 进行原地更新）
                np.maximum.at(xrd_sign, indices, valid_intensities)
        
        # 归一化：将强度值除以100，从[0, 100]缩放到[0, 1]
        xrd_sign = xrd_sign / 100.0
        
        # 空间群标签：转换为0-indexed（row.spacegroup是1-indexed）
        spacegroup_label = row.spacegroup - 1
        
        # 确保标签在有效范围内
        if spacegroup_label < 0 or spacegroup_label >= 230:
            raise ValueError(f"Invalid spacegroup label: {row.spacegroup} (0-indexed: {spacegroup_label})")
        
        return {
            'intensity': xrd_sign,  # numpy array, shape [grid_length], normalized to [0, 1]
            'label': spacegroup_label  # int, 0-229
        }
    
    def get_cls_num_list(self):
        """返回类别分布列表"""
        return self.cls_num_list
    
    def _load_cls_num_list_from_csv(self):
        """从sg_count.csv文件加载类别分布"""
        cls_num_list = [0] * self.num_classes
        
        # 确定sg_count.csv路径
        if self.sg_count_path is None:
            # 尝试从常见位置查找
            possible_paths = [
                '/opt/data/private/ICL/sg_count.csv',
                'sg_count.csv',
                os.path.join(os.path.dirname(__file__), '../../sg_count.csv'),
            ]
            csv_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
        else:
            csv_path = self.sg_count_path
        
        if csv_path is None or not os.path.exists(csv_path):
            print(f"Warning: sg_count.csv not found at {csv_path}, using default uniform distribution")
            # 返回均匀分布作为默认值（基于数据集长度）
            default_count = max(1, self.length // self.num_classes)
            return [default_count] * self.num_classes
        
        # 从CSV文件读取类别分布
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for idx, row in enumerate(reader):
                    if idx >= self.num_classes:
                        break
                    if len(row) >= 2 and row[1].strip():
                        try:
                            count = int(row[1].strip())
                            cls_num_list[idx] = count
                        except ValueError:
                            pass
        except Exception as e:
            print(f"Warning: 读取sg_count.csv失败 ({e})，使用默认分布")
            default_count = max(1, self.length // self.num_classes)
            return [default_count] * self.num_classes
        
        # 确保所有类别都有至少1个样本（避免除零错误）
        cls_num_list = [max(1, count) for count in cls_num_list]
        return cls_num_list
