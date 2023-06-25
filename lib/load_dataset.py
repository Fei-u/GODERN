import os
import numpy as np

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMS03':
        data_path = os.path.join('../data/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0]  
    elif dataset == 'PEMS04':
        data_path = os.path.join('../data/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0]  
    elif dataset == 'PEMS07':
        data_path = os.path.join('../data/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]  
    elif dataset == 'PEMS08':
        data_path = os.path.join('../data/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]  
    elif dataset == 'electricity':
        data_path = os.path.join('../data/electricity.npz')
        data = np.load(data_path)['data'][:, :, 0]  
        print(data.shape)
    elif dataset == 'exchange_rate':
        data_path = os.path.join('../data/exchange_rate.npz')
        data = np.load(data_path)['data'][:, :, 0]  
    elif dataset == 'solar_AL':
        data_path = os.path.join('../data/solar_AL.npz')
        data = np.load(data_path)['data'][:, :, 0]  
    elif dataset == 'traffic':
        data_path = os.path.join('../data/traffic.npz')
        data = np.load(data_path)['data'][:, :, 0] 
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), data.std(),np.median(data))
    return data
