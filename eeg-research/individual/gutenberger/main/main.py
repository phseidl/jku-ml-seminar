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
from datetime import datetime
import yaml
from pathlib import Path
from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig


from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)



def model_select(model_class, model_cfg):
    # model class : ['CLEEGN', 'xLSTM']
    if model_class == 'CLEEGN':
        model = CLEEGN(n_chan=model_cfg['n_chan'], fs=SFREQ, N_F=model_cfg['N_F'])
    elif model_class == 'xLSTM':
        xlstm_cfg = model_cfg['cfg']
        xlstm_cfg = OmegaConf.create(xlstm_cfg)
        xlstm_cfg = from_dict(data_class=xLSTMBlockStackConfig, data=OmegaConf.to_container(xlstm_cfg), config=DaciteConfig(strict=True))
        xlstm_stack = xLSTMBlockStack(xlstm_cfg)
        model = xlstm_stack

    return model


if __name__ == "__main__":
    import argparse

    MODEL_CLASS = 'xLSTM'
    DATASET = 'TUH'    # either 'TUH' or 'BCI'

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if DATASET == 'BCI':
        config_path = 'configs/BCI_KAGGLE/config.json'
        train_anno_path = 'configs/BCI_KAGGLE/set_train.json'
        valid_anno_path = 'configs/BCI_KAGGLE/set_valid.json'
    
    if DATASET == 'TUH':
        config_path = 'configs/TUH/config.yml'
        model_config_path = 'configs/TUH/model_config.yml'
    

    #cfg = read_json(config_path)


    model_name = yaml.safe_load(Path(config_path).read_text())['model_name']
    cfg_dataset = yaml.safe_load(Path(config_path).read_text())['Dataset']
    cfg_general = yaml.safe_load(Path(config_path).read_text())
    cfg_model = yaml.safe_load(Path(model_config_path).read_text())[MODEL_CLASS]

    SFREQ      = cfg_dataset["sfreq"]
    NUM_EPOCHS = cfg_general['epochs']
    BATCH_SIZE = cfg_model['batch_size']
    LR         = cfg_model["learning_rate"]
    

    # Save path
    if cfg_general["save_path"] is None:
        SAVE_PATH = 'logs/' + DATASET + '/' + MODEL_CLASS
        
        if not os.path.exists(SAVE_PATH):
            try:
                os.makedirs(SAVE_PATH)
            except Exception as e:
                print(f"Failed to create directory '{SAVE_PATH}': {e}")
    else:
        SAVE_PATH = cfg_general["save_path"]

    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")

    # Dataset
    x_train, y_train = create_dataset(
        os.path.join(cfg_dataset["x_basepath"], cfg_dataset["x_fpath"]),
        os.path.join(cfg_dataset["y_basepath"], cfg_dataset["y_fpath"]),
        cfg_dataset["subjects_train"], tmin=cfg_dataset["tmin"], tmax=cfg_dataset["tmax"],
        ch_names=cfg_dataset["ch_names"], win_size=cfg_dataset["window_size"], stride=cfg_dataset["stride"]
    )
    x_train = np2TT(np.expand_dims(x_train, axis=1))
    y_train = np2TT(np.expand_dims(y_train, axis=1))

    if MODEL_CLASS == 'xLSTM':
        x_train, y_train = x_train.permute(0,1,3,2).squeeze(), y_train.permute(0,1,3,2).squeeze() 
    
    x_valid, y_valid = create_dataset(
        os.path.join(cfg_dataset["x_basepath"], cfg_dataset["x_fpath"]),
        os.path.join(cfg_dataset["y_basepath"], cfg_dataset["y_fpath"]),
        cfg_dataset["subjects_val"], tmin=cfg_dataset["tmin"], tmax=cfg_dataset["tmax"],
        ch_names=cfg_dataset["ch_names"], win_size=cfg_dataset["window_size"], stride=cfg_dataset["stride"]
    )
    x_valid = np2TT(np.expand_dims(x_valid, axis=1))
    y_valid = np2TT(np.expand_dims(y_valid, axis=1))

    if MODEL_CLASS == 'xLSTM':
        x_valid, y_valid = x_valid.permute(0,1,3,2).squeeze(), y_valid.permute(0,1,3,2).squeeze() 

    print(x_train.size(), y_train.size(), x_valid.size(), y_valid.size())
    
    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    tra_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    validset = torch.utils.data.TensorDataset(x_valid, y_valid)
    val_loader = torch.utils.data.DataLoader(
        validset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )


    model = model_select(MODEL_CLASS, cfg_model).to(device)

    #summary(model, input_size=(BATCH_SIZE, 1, x_train.size()[2], x_train.size()[3]))

    ckpts = [
        Model_Tracer(monitor="loss", mode="min"),
        Model_Tracer(monitor="val_loss", mode="min", do_save=True, root=SAVE_PATH, prefix=model_name + '_' + MODEL_CLASS + '_' + timestamp),
    ]
    criteria = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8, last_epoch=-1)

    tra_time0 = time.time()
    loss_curve = {"epoch": [], "loss": [], "val_loss": []}
    
    for epoch in range(NUM_EPOCHS):  
        loss = train(tra_loader, model, criteria, optimizer)
        
        """ validation """
        val_loss = val(val_loader, model, criteria)
        lr = optimizer.param_groups[-1]['lr']
        optimizer.step()
        #scheduler.step()

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
    #torch.save(model.state_dict(), os.path.join(SAVE_PATH, "model_save_" + timestamp + ".pth"))
### End_Of_File
