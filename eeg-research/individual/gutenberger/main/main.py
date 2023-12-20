from utils.build import create_dataset
from utils.hook import Model_Tracer
from utils.cleegn import CLEEGN
from utils.tv_epoch import train
from utils.tv_epoch import val

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import from_numpy as np2TT
from torchinfo import summary

from os.path import expanduser
from scipy.io import savemat
import numpy as np
import math
import json
import time
import mne
import sys
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_json(filepath):
    if filepath is None:
        return None
    fd = open(filepath, "r")
    content = json.load(fd)
    fd.close()
    return content 

if __name__ == "__main__":
    import argparse
    
    DATASET = 'TUH'    # either 'TUH' or 'BCI'
    
    if DATASET == 'BCI':
        config_path = 'configs/BCI_KAGGLE/config.json'
        train_anno_path = 'configs/BCI_KAGGLE/set_train.json'
        valid_anno_path = 'configs/BCI_KAGGLE/set_valid.json'
    
    if DATASET == 'TUH':
        config_path = 'configs/TUH/config.json'
        train_anno_path = 'configs/TUH/set_train.json'
        valid_anno_path = 'configs/TUH/set_valid.json'
    
    tra_anno = read_json(train_anno_path)
    val_anno = read_json(valid_anno_path)
    cfg = read_json(config_path)

    SFREQ      = tra_anno["sfreq"]
    NUM_EPOCHS = cfg["epochs"]
    BATCH_SIZE = cfg["batch-size"]
    
    if cfg["save-path"] is None:
        SAVE_PATH = os.path.dirname(train_anno_path)
    else:
        SAVE_PATH = cfg["save-path"]

    x_train, y_train = create_dataset(
        os.path.join(tra_anno["x_basepath"], tra_anno["x_fpath"]),
        os.path.join(tra_anno["y_basepath"], tra_anno["y_fpath"]),
        tra_anno["subjects"], tmin=tra_anno["tmin"], tmax=tra_anno["tmax"],
        ch_names=tra_anno["ch_names"], win_size=cfg["window-size"], stride=cfg["stride"]
    )
    x_train = np2TT(np.expand_dims(x_train, axis=1))
    y_train = np2TT(np.expand_dims(y_train, axis=1))
    
    x_valid, y_valid = create_dataset(
        os.path.join(val_anno["x_basepath"], val_anno["x_fpath"]),
        os.path.join(val_anno["y_basepath"], val_anno["y_fpath"]),
        val_anno["subjects"], tmin=val_anno["tmin"], tmax=val_anno["tmax"],
        ch_names=val_anno["ch_names"], win_size=cfg["window-size"], stride=cfg["stride"]
    )
    x_valid = np2TT(np.expand_dims(x_valid, axis=1))
    y_valid = np2TT(np.expand_dims(y_valid, axis=1))
    print(x_train.size(), y_train.size(), x_valid.size(), y_valid.size())
    
    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    tra_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    validset = torch.utils.data.TensorDataset(x_valid, y_valid)
    val_loader = torch.utils.data.DataLoader(
        validset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    model = CLEEGN(n_chan=x_train.size()[2], fs=SFREQ, N_F=x_train.size()[2]).to(device)
    summary(model, input_size=(BATCH_SIZE, 1, x_train.size()[2], x_train.size()[3]))

    ckpts = [
        Model_Tracer(monitor="loss", mode="min"),
        Model_Tracer(monitor="val_loss", mode="min", do_save=True, root=SAVE_PATH, prefix=cfg["model_name"]),
    ]
    criteria = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8, last_epoch=-1)

    tra_time0 = time.time()
    loss_curve = {"epoch": [], "loss": [], "val_loss": []}
    
    for epoch in range(NUM_EPOCHS):  
        loss = train(tra_loader, model, criteria, optimizer)
        
        """ validation """
        val_loss = val(val_loader, model, criteria)
        lr = optimizer.param_groups[-1]['lr']
        optimizer.step()
        scheduler.step()

        print("\rEpoch {}/{} - {:.2f} s - loss: {:.4f} - val_loss: {:.4f} - lr: {:e}".format(
            epoch + 1, NUM_EPOCHS, time.time() - tra_time0, loss, val_loss, lr
        ))
        state = dict(
            epoch=epoch + 1, min_loss=ckpts[0].bound, min_vloss=ckpts[1].bound,
            state_dict=model.state_dict(), loss=loss, val_loss=val_loss, learning_rate=lr
        )
        for ckpt in ckpts:
            ckpt.on_epoch_end(epoch + 1, state)
        loss_curve["epoch"].append(epoch + 1)
        loss_curve["loss"].append(loss)
        loss_curve["val_loss"].append(val_loss)
    ### End_Of_Train
    savemat(os.path.join(SAVE_PATH, "loss_curve.mat"), loss_curve)
### End_Of_File
