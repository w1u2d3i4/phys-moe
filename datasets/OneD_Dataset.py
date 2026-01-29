from torch import multiprocessing 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch 
import numpy as np
import os
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from collections import defaultdict

TO_XRD_LENGTH = 8500
ANGLE_START = 5 
ANGLE_END = 90

class XrdData(Dataset):
    cls_num = 230
    def __init__(self,file_path):
        self.num_classes = 230
        data = np.load(file_path,allow_pickle=True,encoding='latin1')
        #print(data)
        self.intensity = data.item().get('intensitys')
        self.features = data.item().get('features')
        self.angle = data.item().get('angles')
        #self.labels230 = data.item().get('labels230')
        self.targets = data.item().get('labels230')
        self.labels7 = data.item().get('labels7')

        #修改labels230和labels7，转换为np.array
        #self.labels230 = np.atleast_1d(self.labels230)
        #self.labels7 = np.atleast_1d(self.labels7)

        self.lattice = data.item().get('lattices')
        self.atomic_labels = data.item().get('atomic_labels')
        self.mask = data.item().get('mask')
        self.cart_coords = data.item().get('cart_coords')
        if self.intensity is None:
            self.intensity = data.item().get('features')
        if self.angle is None:
            self.angle = np.arange(ANGLE_START,ANGLE_END,(ANGLE_END-ANGLE_START)/TO_XRD_LENGTH).reshape(1,-1).repeat(len(self.targets),axis=0).astype(np.float32)
        #print(file_path,len(file_path))
        #if len(file_path)<= 55:
        #    self.intensity = np.sum(self.intensity.squeeze(),axis=1)
        self.idx = np.arange(0,TO_XRD_LENGTH)
        # atomic_number  = data.item().get('atomic_number')
        # print(self.features.shape,self.angle.shape,self.labels230.shape,self.labels7.shape)
        # Calculate class distribution for get_cls_num_list
        self.cls_num_list = self._get_cls_num_list()

    def __getitem__(self, index):
        #return [self.intensity[index],self.angle[index],self.labels230[index],self.idx]
        # 归一化：将强度值除以100，从[0, 100]缩放到[0, 1]
        intensity = self.intensity[index] / 100.0
        return {'intensity': intensity, 'label': self.targets[index]}
        # torch.Size([16, 850]) torch.Size([16, 850]) torch.Size([16]) torch.Size([16, 3, 3]) torch.Size([16, 500]) torch.Size([16, 500]) torch.Size([16, 500, 3])
        # return [self.intensity[index],
        #         self.angle[index],
        #         self.labels230[index],
        #         self.index,
        #         sp2cs[self.labels230[index]],
        #         sp2lt[self.labels230[index]],
        #         sp2pg[self.labels230[index]],
        #         ]
                # self.lattice[index],
                # self.atomic_labels[index],
                # self.mask[index],
                # self.cart_coords[index]]

    def __len__(self):
        return len(self.targets)
    
    def get_cls_num_list(self):
        return self.cls_num_list
    
    def _get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.num_classes):
            cls_num_list.append(np.sum(self.targets == i))
        return cls_num_list


class OneDDataset(Dataset):
    cls_num = 230
    
    def __init__(self, args, train=True):
        self.args = args
        self.train = train
        self.num_classes = 230
        self.data_len = 8500
        
        # TODO: Load your real data here
        # Example:
        # self.data = np.load('path/to/data.npy')
        # self.labels = np.load('path/to/labels.npy')
        
        # For now, generating dummy data
        num_samples = 1000 if train else 200
        self.data = np.random.randn(num_samples, self.data_len).astype(np.float32)
        self.labels = np.random.randint(0, self.num_classes, size=num_samples)
        
        # Calculate class distribution for get_cls_num_list
        self.cls_num_list = self._get_cls_num_list()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Return data as tensor. 
        # Note: Model expects (N, 1, L) or (N, L). 
        # If model handles unsqueeze, we can return (L,).
        # Resnet1D_MoE forward does unsqueeze if dim is 2 (batch, length).
        # So here we return (length,).
        
        sample = self.data[index]
        label = self.labels[index]
        
        return {'image': torch.from_numpy(sample), 'label': torch.tensor(label, dtype=torch.long)}

    def get_cls_num_list(self):
        return self.cls_num_list

    def _get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.num_classes):
            cls_num_list.append(np.sum(self.labels == i))
        return cls_num_list

# Alias for validation set if needed, or just use same class
class OneDDatasetVal(OneDDataset):
    def __init__(self, args, train=False):
        super().__init__(args, train=train)

