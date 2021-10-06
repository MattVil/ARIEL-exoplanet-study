import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from model import ArielNetwork, ResidualNet

# DATA_DIR = "I:\CNES\ml_data_challenge_database\ml_data_challenge_database"
DATA_DIR = "/media/matthieu/8917c2f9-55b5-458f-b893-826637dda6e6/CNES"

def load_files_names(file_name):
    f_noisy_train = []
    with open(os.path.join(DATA_DIR, file_name)) as f:
        for line in f.readlines():
            f_noisy_train.append(line.replace('\n', ''))
    return f_noisy_train

def get_file(files_list, idx):
    infos = {}
    data = []
    with open(os.path.join(DATA_DIR, files_list[idx])) as f:
        for i, line in enumerate(f.readlines()):
            if i < 6:
                infos[line.split(':')[0][2:]] = float(line.split(':')[1][1:])
            else:
                data.append(line.split('\t'))
    infos['file_name'] = files_list[idx]
    return infos, np.array(data).astype(float)

def plot_spectrogram(infos, data):
    ax = plt.subplot()
    im = ax.imshow(data)
    plt.xlabel("Time")
    plt.ylabel("Wavelength channel")
    plt.title("Sample : {}\nStar_temp: {:.2f} Star_logg: {:.2f} Star_rad: {:.2f} Star_mass: {:.2f} Period: {:.2f}".format(
        infos['file_name'], infos['star_temp'], infos['star_logg'], infos['star_rad'], infos['star_mass'], infos['period']
    ))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()

def plot_wavelength(infos, data):
    
    plt.plot(data[0])
    plt.plot(data[1])
    plt.plot(data[2])
    # plt.plot(data[3])
    # plt.plot(data[4])
    plt.ylim(0.98, 1.02)  
    plt.show()

if __name__ == '__main__':
    f_noisy_train = load_files_names('noisy_train.txt')
    infos, data = get_file(f_noisy_train, 1331)
    print(data.shape)
    print(infos)
    plot_wavelength(infos, data)
    # plot_spectrogram(infos, data)
    # data_torch = torch.tensor(data).unsqueeze(0).unsqueeze(0).type(torch.cuda.FloatTensor)
    # print(data_torch.shape)
    # model = ArielNetwork(128, 8).cuda()
    # data_conv = model(data_torch)
    # print(data_conv.shape)