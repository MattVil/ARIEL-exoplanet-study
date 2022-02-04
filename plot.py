from turtle import color
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from torch.utils.data import Dataset, DataLoader

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
  keys = ['star_temp', 'star_logg', 'star_rad', 'star_mass', 'star_k_mag', 'period', 'sma', 'incl']
  # keys.remove('file_name')

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
              "Mean: {:.1f}\nMin: {:.1f}\nMax: {:.1f}\nStd: {:.1f}".format(data[key].mean(), data[key].min(), data[key].max(), data[key].std()),
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
    
def plot_light_curves(data, y):
  T, W = data.shape
  data_df = pd.DataFrame(data=data)
  y_df = pd.DataFrame(data=np.tile(y, (T, 1)))
  mean_data_df = data_df.mean(axis=1)
  std_data_df = data_df.std(axis=1)
  low = mean_data_df - std_data_df
  high = mean_data_df + std_data_df
  plt.errorbar(np.arange(T), mean_data_df, yerr=std_data_df, alpha=.5, fmt='-', linewidth=2, elinewidth=0.5, capsize=2, capthick=0.5)
  plt.fill_between(x=np.arange(T), y1=low, y2=high, alpha=.25)
  plt.ylim(low.min(), high.max())
  plt.title('Mean lightcurve. Mean Target : {:.4f}'.format(np.mean(y)))
  plt.show()
  
if __name__ == '__main__':
  sns.set_theme(style="whitegrid")
  
  print('Loading data ...')
  dataset = ArielDataset(DATA_DIR, 'train')
  dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
  print('Done.')
  for i_batch, sample_batched in enumerate(dataloader):
      X = sample_batched['data'].squeeze(0).permute(1, 0).cpu().detach().numpy()
      y = sample_batched['labels'].squeeze(0).cpu().detach().numpy()
      plot_light_curves(X[25:-25], y)

    

    
       
