import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader

from model import ArielCNN

from main import plot_spectrogram
DATA_DIR = "I:\CNES\ml_data_challenge_database\ml_data_challenge_database"
# DATA_DIR = "/media/matthieu/8917c2f9-55b5-458f-b893-826637dda6e6/CNES"
MAX_TRAIN = 100000
MAX_EVAL = 10000


class ArielDataset(Dataset):
    
    def __init__(self, data_dir, mode):
        assert mode in ['train', 'eval', 'test']
        
        self.data_dir = data_dir
        self.mode = mode
        if mode == 'train':
            self.file_names = self.__load_files_names('noisy_train_t.txt', 'train')
        elif mode == 'eval':
            self.file_names = self.__load_files_names('noisy_train_v.txt', 'eval')
        elif mode == 'test':
            self.file_names = self.__load_files_names('noisy_test.txt', 'test')
        # random.shuffle(self.file_names)
    
    def __load_files_names(self, file_name, mode):
        f_names = []
        with open(os.path.join(self.data_dir, file_name)) as f:
            for line in f.readlines():
                f_names.append(line.replace('\n', ''))
        if mode == 'train':
            return f_names[:MAX_TRAIN]
        elif mode == 'eval':
            return f_names[:MAX_EVAL]
        return f_names
        
    def __len__(self):
        return len(self.file_names)
    
    def __normalize_infos(self, data, mode='min_max'):
        min = torch.tensor([ 3103.7, 3.5, 0.1, 0.1,  1.8,  0.3])
        max = torch.tensor([ 6995.7, 5.0, 3.3, 2.2, 13.5, 68.1])
        mean = torch.tensor([4892.7, 4.6, 0.8, 0.8, 10.5,  6.5])
        std = torch.tensor([  982.5, 0.3, 0.4, 0.3,  1.6,  6.0])
        
        if mode == 'min_max':
            return (data-min)/(max-min)
        elif mode == 'mean_std':
            return (data-mean)/std
        else:
            return None
    
    def __getitem__(self, idx):
        infos = {}
        data = []
        params = []
        labels = []
        aux_labels = []
        file_id = self.file_names[idx].split('/')[1]
        with open(os.path.join(self.data_dir, self.file_names[idx])) as f:
            for i, line in enumerate(f.readlines()):
                if i < 6:
                    infos[line.split(':')[0][2:]] = float(line.split(':')[1][1:])
                    params.append(float(line.split(':')[1][1:]))
                else:
                    data.append(line.replace('\n', '').split('\t'))
        if self.mode in ['train', 'eval']:
            with open(os.path.join(self.data_dir, 'params_train', file_id)) as f:
                for i, line in enumerate(f.readlines()):
                    if i < 2:
                        aux_labels.append(float(line.split(':')[1][1:]))
                        infos[line.split(':')[0][2:]] = float(line.split(':')[1][1:])
                    else:
                        labels.append(line.replace('\n', '').split('\t'))
        infos['file_name'] = file_id
        data_infos = self.__normalize_infos(torch.tensor([infos['star_temp'], infos['star_logg'], infos['star_rad'], infos['star_mass'], infos['star_k_mag'], infos['period']]), mode='mean_std')
        return {'infos': infos , 
                'data': torch.tensor(np.array(data).astype(float)).type(torch.FloatTensor).cuda(), 
                'data_infos' : data_infos.type(torch.FloatTensor).cuda(),
                'params': torch.tensor(np.array(params).astype(float)).type(torch.FloatTensor).cuda(), 
                'labels': torch.tensor(np.array(labels).astype(float)).squeeze(0).type(torch.FloatTensor).cuda(), 
                'aux_labels': torch.tensor(np.array(aux_labels).astype(float)).type(torch.FloatTensor).cuda()}
    
if __name__ == '__main__':
    model = ArielCNN().cuda()
    dataset = ArielDataset(DATA_DIR, 'train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    for i_batch, sample_batched in enumerate(dataloader):
        print(sample_batched['infos'], sample_batched['data'].shape, sample_batched['labels'].shape, sample_batched['aux_labels'].shape)
        X = sample_batched['data']
        X_aux = sample_batched['params']
        infos = sample_batched['infos']
        print(X.shape, X_aux.shape)
        y, y_aux = model(X, X_aux)
        print(y.shape, y_aux.shape)
        data2 = sample_batched['data_infos']
        print(data2.shape, data2)
        break