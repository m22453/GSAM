import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as scio
from sklearn import preprocessing


# for multi-view document datasets
class MultiViewDataset(Dataset):
    def __init__(self, embedding_list, labels):
        super(MultiViewDataset, self).__init__()
        self.embedding_list = embedding_list
        self.labels = labels

    def __getitem__(self, index):
        data = []
        for i in range(len(self.embedding_list)):
            data.append(np.array(self.embedding_list[i][index], dtype=np.float))
        label = self.labels[index]
        return data, label

    def __len__(self):
        return len(self.labels)
    
class MultiViewImgDataset(Dataset):
    def __init__(self, embedding_list, labels):
        super(MultiViewImgDataset, self).__init__()
        self.embedding_list = embedding_list
        self.labels = labels

    def __getitem__(self, index):
        data = []
        for i in range(len(self.embedding_list)):
            data.append(np.array(self.embedding_list[i][index].detach().cpu().numpy(), dtype=np.float))
        label = self.labels[index]
        return data, label

    def __len__(self):
        return len(self.labels)


# for multi-view image dataset
class UCIDataset(Dataset):
    def __init__(self, data_path='./data/uci-digit.mat'):
        super(UCIDataset, self).__init__()

        print('dataset_name:', data_path.split('/')[-1])
        data = scio.loadmat(data_path)
        # print(data)
        self.x1 = data['mfeat_fac']/1.0
        print('mfeat_fac shape', self.x1.shape)
        self.x2 = data['mfeat_fou']/1.0
        print('mfeat_fou shape', self.x2.shape)
        self.x3 = data['mfeat_kar']/1.0
        print('mfeat_kar shape', self.x3.shape)


        standardScaler = preprocessing.StandardScaler()

        self.x1 = standardScaler.fit_transform(self.x1)
        self.x2 = standardScaler.fit_transform(self.x2)
        self.x3 = standardScaler.fit_transform(self.x3)
        # self.x3 = min_max_scaler.fit_transform(self.x3)


        y = data['truth']
        print(y.shape)
        self.y = y.reshape(y.shape[0])


        self.x = [self.x1, self.x2, self.x3]
        self.x_n = [self.x1.shape[-1], self.x2.shape[-1], self.x3.shape[-1]]

        self.input_dim = max(self.x_n)
        self.size = len(self.y)
        self.v_n = 3
        self.cluster_n = len(np.unique(self.y))
        

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data = [torch.tensor(self.x1[idx], dtype=torch.float32), torch.tensor(self.x2[idx], dtype=torch.float32), 
        torch.tensor(self.x3[idx], dtype=torch.float32)]
        label = self.y[idx]
        return data, label

    
class S15Datasets(Dataset):
    def __init__(self, data_path='./data/Scene15.mat'):
        super(S15Datasets, self).__init__()

        print('dataset_name:', data_path.split('/')[-1])
        data = scio.loadmat(data_path)
        self.x1 = data['X'][0][0]
        self.x2 = data['X'][0][1]

        standardScaler = preprocessing.StandardScaler()
        self.x1 = standardScaler.fit_transform(self.x1)
        self.x2 = standardScaler.fit_transform(self.x2)
        # StandardScaler


        print(self.x1.shape)
        print(self.x2.shape)
        y = data['Y']
        print(y.shape)
        self.y = y.reshape(y.shape[0])


        self.x = [self.x1, self.x2]
        self.x_n = [self.x1.shape[-1], self.x2.shape[-1]]

        self.input_dim = max(self.x_n)
        self.size = len(self.y)
        self.v_n = 2
        self.cluster_n = len(np.unique(self.y))
        

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data = [torch.tensor(self.x1[idx], dtype=torch.float32), torch.tensor(self.x2[idx], dtype=torch.float32)]
        label = self.y[idx]
        return data, label