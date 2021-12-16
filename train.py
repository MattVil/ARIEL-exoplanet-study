import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from statistics import mean

from model import ArielCNN, ArielNetwork, CrossAttentionNet
from data import ArielDataset

torch.manual_seed(0)

DATA_DIR = "I:\CNES\ml_data_challenge_database\ml_data_challenge_database"
# DATA_DIR = "/media/matthieu/8917c2f9-55b5-458f-b893-826637dda6e6/CNES"

def eval(model, dataloader, loss_fn):
    model.eval()
    losses = []
    with torch.no_grad():
        for i_batch, sample_batched in tqdm(enumerate(dataloader)):
            X = sample_batched['data'] 
            X_param = sample_batched['params']
            Y = sample_batched['labels']
            Y_aux = sample_batched['aux_labels']

            pred = model(X)
            loss_value = loss_fn(pred, Y)
            losses.append(loss_value)
    
    model.train()
    return torch.mean(torch.stack(losses))
    
def main():
    # loss_fn = nn.MSELoss()
    loss_fn = nn.SmoothL1Loss()
    # model = ArielNetwork(128, 8).cuda()
    # model = ArielCNN().cuda()
    model = CrossAttentionNet().cuda()

    optimizer = AdamW(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    dataset_train = ArielDataset(DATA_DIR, 'train')
    dataLoader = DataLoader(dataset_train, batch_size=32, shuffle=True)

    dataset_eval = ArielDataset(DATA_DIR, 'eval')
    dataLoader_eval = DataLoader(dataset_eval, batch_size=32, shuffle=False)

    dataset_test = ArielDataset(DATA_DIR, 'test')
    dataLoader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

    print("{} Training samples ({})".format(len(dataset_train), len(dataLoader)))
    print("{} Validation samples ({})".format(len(dataset_eval), len(dataLoader_eval)))
    print("{} Test samples ({})".format(len(dataset_test), len(dataLoader_test)))


    EPOCHS = 10
    log_every = 1

    train_loss, eval_loss = [], []

    for epoch in range(EPOCHS):
        print("#"*20+"\nEpoch {}/{}".format(epoch+1, EPOCHS))
        t0 = time.time()
        loss_log = []
        for i_batch, sample_batched in tqdm(enumerate(dataLoader)):
            X = sample_batched['data'] 
            X_param = sample_batched['params']
            Y = sample_batched['labels']
            Y_aux = sample_batched['aux_labels']

            pred = model(X)
            loss_value = loss_fn(pred, Y)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            loss_log.append(loss_value)

        train_loss.append(torch.mean(torch.stack(loss_log)))
        eval_loss_value = eval(model, dataLoader_eval, loss_fn=loss_fn)
        eval_loss.append(eval_loss_value)
        print("Training Loss: {}\tEvaluation Loss: {}".format(train_loss[-1], eval_loss[-1]))
        print('Time : {:.2f}min'.format((time.time()-t0)/60))
        t0 = time.time()
        print(scheduler.get_last_lr())
        scheduler.step()
        torch.save(model.state_dict(), 'models/model{}.pt'.format(epoch))

        train_loss_ = torch.stack(train_loss).cpu().detach().numpy()
        eval_loss_ = torch.stack(eval_loss).cpu().detach().numpy()
        with open('train_loss.npy', 'wb') as f:
            np.save(f, train_loss_)
        with open('eval_loss.npy', 'wb') as f:
            np.save(f, eval_loss_)
    # plt.plot(train_loss[:, 0], '--', label="Total training loss", color='b')
    plt.plot(train_loss, label="training loss", color='b')
    # plt.plot(train_loss[:, 2], ':', label="Aux training loss", color='b')
    # plt.plot(eval_loss[:, 0], '--', label="Total validation loss", color='g')
    plt.plot(eval_loss, label="validation loss", color='g')
    plt.legend()
    plt.savefig("imgs/loss.png")
    # plt.plot(eval_loss[:, 2], ':', label="Aux validation loss", color='g')
    plt.show()

    print("EVALUATION")
    losses = []
    with open('my_predictions.txt', 'w') as f:
        for i_batch, sample_batched in tqdm(enumerate(dataLoader_test)):
            f_name = sample_batched['infos']['file_name']
            X = sample_batched['data'] 
            X_param = sample_batched['params']

            pred = model(X)

            line = f_name[0]
            pred_np = pred.cpu().detach().numpy()[0]

            for res in pred_np:
                line += "\t{}".format(res)
            line += '\n'
            f.write(line)
            
if __name__ == '__main__':
    # torch.multiprocessing.set_start_method("spawn")
    main()