import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from statistics import mean

from model import ArielCNN, ArielNetwork
from data import ArielDataset

torch.manual_seed(0)

DATA_DIR = "/media/matthieu/8917c2f9-55b5-458f-b893-826637dda6e6/CNES"

def eval(model, dataloader):
    model.eval()
    losses = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            X = sample_batched['data'] 
            X_param = sample_batched['params']
            Y = sample_batched['labels']
            Y_aux = sample_batched['aux_labels']

            pred, aux_pred = model(X, X_param)
            loss_value = loss_fn(pred, Y)
            loss_value_aux = loss_fn(aux_pred, Y_aux)
            loss_value_sum = loss_value + loss_value_aux
            losses.append([
                float(loss_value_sum.cpu().detach().numpy()),
                float(loss_value.cpu().detach().numpy()), 
                float(loss_value_aux.cpu().detach().numpy())
            ])
    
    model.train()
    losses = np.array(losses)
    return [losses[:, 0].mean(), losses[:, 1].mean(), losses[:, 2].mean()]
    

loss_fn = nn.MSELoss()
# model = ArielNetwork(128, 8).cuda()
model = ArielCNN().cuda()
optimizer = AdamW(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

dataset_train = ArielDataset(DATA_DIR, 'train')
dataLoader = DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=0)

dataset_eval = ArielDataset(DATA_DIR, 'eval')
dataLoader_eval = DataLoader(dataset_eval, batch_size=1, shuffle=False, num_workers=0)

dataset_test = ArielDataset(DATA_DIR, 'test')
dataLoader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)

print("{} Training samples".format(len(dataset_train)))
print("{} Validation samples".format(len(dataLoader_eval)))
print("{} Test samples".format(len(dataLoader_test)))


EPOCHS = 10
log_every = 1

train_loss, eval_loss = [], []

for epoch in range(EPOCHS):
    print("Epoch {}/{}".format(epoch+1, EPOCHS))
    t0 = time.time()
    loss_log = []
    for i_batch, sample_batched in enumerate(dataLoader):
        X = sample_batched['data'] 
        X_param = sample_batched['params']
        Y = sample_batched['labels']
        Y_aux = sample_batched['aux_labels']

        pred, aux_pred = model(X, X_param)
        loss_value = loss_fn(pred, Y)
        loss_value_aux = loss_fn(aux_pred, Y_aux)
        loss_value_sum = loss_value + loss_value_aux


        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        loss_log.append([
            float(loss_value_sum.cpu().detach().numpy()),
            float(loss_value.cpu().detach().numpy()), 
            float(loss_value_aux.cpu().detach().numpy())
        ])
        if i_batch % log_every == 0:
            loss_log = np.array(loss_log)
            train_loss.append([loss_log[:, 0].mean(), loss_log[:, 1].mean(), loss_log[:, 2].mean()])
            loss_log = []
            eval_loss_value = eval(model, dataLoader_eval)
            eval_loss.append(eval_loss_value)
            print("Training Loss: {}\tEvaluation Loss: {}".format(train_loss[-1][1], eval_loss[-1][1]))
    print('Time : {:.2f}min'.format((time.time()-t0)/60))
    t0 = time.time()
    scheduler.step()
    print(scheduler.get_last_lr())

torch.save(model.state_dict(), 'my_model.pt')

train_loss = np.array(train_loss)
eval_loss = np.array(eval_loss)
# plt.plot(train_loss[:, 0], '--', label="Total training loss", color='b')
plt.plot(train_loss[:, 1], label="Task training loss", color='b')
# plt.plot(train_loss[:, 2], ':', label="Aux training loss", color='b')
# plt.plot(eval_loss[:, 0], '--', label="Total validation loss", color='g')
plt.plot(eval_loss[:, 1], label="Task validation loss", color='g')
# plt.plot(eval_loss[:, 2], ':', label="Aux validation loss", color='g')
plt.show()

with open('my_predictions.txt', 'w') as f:
    for i_batch, sample_batched in enumerate(dataLoader_test):
        f_name = sample_batched['infos']['file_name']
        X = sample_batched['data'] 
        X_param = sample_batched['params']

        pred, aux_pred = model(X, X_param)

        line = f_name[0]
        pred_np = pred.cpu().detach().numpy()[0]

        for res in pred_np:
            line += "\t{}".format(res)
        line += '\n'
        f.write(line)