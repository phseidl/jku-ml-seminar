from utils.build import create_dataset
from utils.hook import Model_Tracer
from utils.cleegn import CLEEGN
from utils.seq2seq import Seq2Seq, Seq2SeqLSTM, LSTM
from utils.seq2seq_attention import Seq2SeqWithAttention
from utils.lstm_autoencoder import LSTMAutoencoder
from utils.seq2seq_transformer import TransformerDenoiser
from utils.Autoencoder_CNN import ConvAutoencoder, ConvAutoencoder_Compress
from utils.Autoencoder_CNN_LSTM import LSTMConvAutoencoder, LSTMConvAutoencoder2, LSTMConvAutoencoder3, LSTMConvAutoencoder4
from utils.Parallel_CNN_LSTM import Parallel_CNN_LSTM
from utils.OneD_ResCNN import OneD_ResCNN
from utils.IC_U_NET import IC_U_NET
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
import wandb
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

USE_WANDB = True
ensemble_loss = True

def model_select(model_class, model_cfg):
    if model_class == 'CLEEGN':
        model = CLEEGN(n_chan=model_cfg['n_chan'], fs=128, N_F=model_cfg['N_F'])
    elif model_class == 'xLSTM':
        xlstm_cfg = model_cfg['cfg']
        xlstm_cfg = OmegaConf.create(xlstm_cfg)
        xlstm_cfg = from_dict(data_class=xLSTMBlockStackConfig, data=OmegaConf.to_container(xlstm_cfg), config=DaciteConfig(strict=True))
        xlstm_stack = xLSTMBlockStack(xlstm_cfg)
        model = xlstm_stack
    elif model_class == 'Seq2Seq':
        model = Seq2Seq(input_dim=model_cfg['n_chan'], hidden_dim=model_cfg['hidden_dim'], output_dim=model_cfg['n_chan'], num_layers=model_cfg['num_layers'])
    elif model_class == 'Seq2Seq_LSTM':
        model = Seq2SeqLSTM(input_dim=model_cfg['n_chan'], hidden_dim=model_cfg['hidden_dim'], output_dim=model_cfg['n_chan'], num_layers=model_cfg['num_layers'])
    elif model_class == 'LSTM':
        model = LSTM(input_dim=model_cfg['n_chan'], hidden_dim=model_cfg['hidden_dim'], output_dim=model_cfg['n_chan'], num_layers=model_cfg['num_layers'])
    elif model_class == 'Seq2Seq_Attention':
        model = Seq2SeqWithAttention(input_dim=model_cfg['n_chan'], hidden_dim=model_cfg['hidden_dim'], output_dim=model_cfg['n_chan'], num_layers=model_cfg['num_layers'])
    elif model_class == 'LSTM_Autoencoder':
        model = LSTMAutoencoder(input_dim=model_cfg['n_chan'], hidden_dim=model_cfg['hidden_dim'], latent_dim=model_cfg['latent_dim'], output_dim=model_cfg['n_chan'], num_layers=model_cfg['num_layers'])
    elif model_class == 'Transformer':
        model = TransformerDenoiser(input_dim=model_cfg['n_chan'], embed_dim = model_cfg['embed_dim'], num_heads = model_cfg['num_heads'], num_layers = model_cfg['num_layers'], hidden_dim = model_cfg['hidden_dim'], dropout = model_cfg['dropout'], max_len = model_cfg['max_len'])
    elif model_class == 'Autoencoder_CNN':
        model = ConvAutoencoder(input_channels=model_cfg['n_chan'])
    elif model_class == 'Autoencoder_CNN_Compress':
        model = ConvAutoencoder_Compress(input_channels=model_cfg['n_chan'])
    elif model_class == 'Autoencoder_CNN_LSTM':
        model = LSTMConvAutoencoder(input_channels=model_cfg['n_chan'], sequence_length=model_cfg['sequence_length'], hidden_dim=model_cfg['hidden_dim'], latent_dim=model_cfg['latent_dim'])
    elif model_class == 'Autoencoder_CNN_LSTM2':
        model = LSTMConvAutoencoder2(input_dim=model_cfg['n_chan'], hidden_dim = model_cfg['n_chan'], num_layers = model_cfg['num_layers'])
    elif model_class == 'Autoencoder_CNN_LSTM3':
        model = LSTMConvAutoencoder3(input_dim=model_cfg['n_chan'], num_layers = model_cfg['num_layers'])
    elif model_class == 'Autoencoder_CNN_LSTM4':
        model = LSTMConvAutoencoder4(input_dim=model_cfg['n_chan'], num_layers = model_cfg['num_layers'])        
    elif model_class == 'Parallel_CNN_LSTM':
        model = Parallel_CNN_LSTM(lstm_model=LSTM(input_dim=model_cfg['n_chan'], hidden_dim=model_cfg['n_chan'], output_dim=model_cfg['n_chan'], num_layers=model_cfg['num_layers']), cnn_model=ConvAutoencoder(input_channels=model_cfg['n_chan']), n_chan=model_cfg['n_chan'], learn_concat=model_cfg['learn_concat'])
    elif model_class == 'IC_U_Net':
        #model = IC_U_NET(n_channels=model_cfg['n_chan'], bilinear=model_cfg['bilinear'])
        model = IC_U_NET(input_channels=model_cfg['n_chan'])
    elif model_class == 'OneD_Res_CNN':
        model = OneD_ResCNN(seq_length=model_cfg['seq_length'], batch_size=model_cfg['batch_size'], n_chan=model_cfg['n_chan'])
    return model


def main_fct(config = None):
    import argparse
    
    if USE_WANDB:
        wandb.init(config=config)
        config = wandb.config
        MODEL_CLASS = config.model_class
    else:   
        MODEL_CLASS = config['model_class']

    DATASET = 'TUH'    # either 'TUH' or 'BCI'

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if DATASET == 'BCI':
        config_path = 'configs/BCI_KAGGLE/config.yml'
        model_config_path = 'configs/BCI_KAGGLE/model_config.yml'
    
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
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    validset = torch.utils.data.TensorDataset(x_valid, y_valid)
    val_loader = torch.utils.data.DataLoader(
        validset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )


    model = model_select(MODEL_CLASS, cfg_model).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print ('Total # of modelparameters: ', str(total_params))

    #summary(model, input_size=(BATCH_SIZE, 1, x_train.size()[2], x_train.size()[3]))

    ckpts = [
        Model_Tracer(monitor="loss", mode="min"),
        Model_Tracer(monitor="val_loss", mode="min", do_save=True, root=SAVE_PATH, prefix= MODEL_CLASS + '_' + timestamp),
    ]
    criteria = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    #optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8, last_epoch=-1)

    tra_time0 = time.time()
    loss_curve = {"epoch": [], "loss": [], "val_loss": []}
    
    
    for epoch in range(NUM_EPOCHS):  
        loss = train(tra_loader, model, criteria, optimizer, MODEL_CLASS, ensemble_loss, USE_WANDB)
        
        """ validation """
        val_loss = val(val_loader, model, criteria, MODEL_CLASS)
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
        if USE_WANDB:
            wandb.log({"epoch": epoch, "loss": loss, "val_loss": val_loss})
        loss_curve["epoch"].append(epoch + 1)
        loss_curve["loss"].append(loss)
        loss_curve["val_loss"].append(val_loss)
    ### End_Of_Train
    savemat(os.path.join(SAVE_PATH, "loss_curve.mat"), loss_curve)

if __name__ == "__main__":
    if USE_WANDB:
        wandb.login()

        sweep_config = {
                'method': 'grid',
            }

        parameters_dict = {
        'model_class': {
            'values': ['IC_U_Net'] #['LSTM', 'Autoencoder_CNN', 'xLSTM', 'Autoencoder_CNN_LSTM2', 'Autoencoder_CNN_LSTM3', 'Autoencoder_CNN_LSTM4', 'Parallel_CNN_LSTM', 'CLEEGN', 'IC_U_Net', 'OneD_Res_CNN']
            },
        }

        sweep_config['parameters'] = parameters_dict

        sweep_id = wandb.sweep(sweep_config, project="EEG_Denoising")

        wandb.agent(sweep_id, main_fct)
    else:
        config = {
        'model_class': 'IC_U_Net' #['LSTM', 'Autoencoder_CNN', 'xLSTM', 'Autoencoder_CNN_LSTM2', 'Autoencoder_CNN_LSTM3', 'Autoencoder_CNN_LSTM4', 'Parallel_CNN_LSTM', 'CLEEGN', 'IC_U_Net', 'OneD_Res_CNN']
            }
        main_fct(config)