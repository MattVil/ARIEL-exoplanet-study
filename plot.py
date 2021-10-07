import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data import ArielDataset
from data import DATA_DIR

def load_all_data(dataset):
    dataset_values = {}
    for key in dataset[0]['infos'].keys():
        dataset_values[key] = []

    dataset_values['data'] = []
    dataset_values['labels'] = []

    for data in tqdm(dataset):
        for key in data['infos'].keys():
            dataset_values[key].append(data['infos'][key])
        dataset_values['data'].append(data['data'].cpu().detach().numpy())
        dataset_values['labels'].append(data['labels'].cpu().detach().numpy())
    for key, values in dataset_values.items():
        dataset_values[key] = np.array(values)

    return dataset_values

def plot_dataset_infos(data):

    fig, axs = plt.subplots(1, 8)
    fig.suptitle('Ariel Dataset Analysis : Test set')
    
    keys = list(data.keys())
    keys.remove('file_name')

    for i, key in enumerate(keys):
        bar = axs[i].bar(
            key, 
            data[key].mean(), 
            yerr=data[key].std(), 
            align='center', alpha=0.5, ecolor='black', capsize=10
        )
        for idx, rect in enumerate(bar):
            height = rect.get_height()
            axs[i].text(rect.get_x() + rect.get_width()/2., 0.25*height,
                    "Mean: {:.1f}\nMin: {:.1f}\nMax: {:.1f}".format(data[key].mean(), data[key].min(), data[key].max()),
                    ha='center', va='bottom')
        
    # fig.tight_layout(pad=0.01)
    plt.show()

def plot_labels_distrib(data):
    mean = np.mean(data['labels'], axis=0)
    std = np.std(data['labels'], axis=0)
    arr = np.arange(55)
    print(mean.shape, std.shape)
    fig, ax = plt.subplots()
    bar = plt.bar(
        arr, 
        mean, 
        yerr=std, 
        align='center', alpha=0.5, ecolor='black', capsize=10
    )
    for idx, rect in enumerate(bar):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 0.1*height,
                "Min: {:.2f} Max: {:.2f}".format(data['labels'][:,idx].min(), data['labels'][:,idx].max()),
                ha='center', va='bottom', rotation=90)

    plt.xlabel('wavelength')
    plt.ylabel('mean value')
    plt.show()

if __name__ == '__main__':
    dataset = ArielDataset(DATA_DIR, 'train')
    data = load_all_data(dataset)
    plot_labels_distrib(data)

    

    
       
