import torch
import torch.nn as nn
from torch import from_numpy as np2TT
from scipy.io import savemat
import numpy as np
import time
import copy
import os
import wandb
from datetime import datetime
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

from BaselineModels.utils.misc import model_select
from BaselineModels.utils.build import create_dataset, create_EEG_DenoiseNet_dataset
from UPT4EEG.utils.model_tracer import Model_Tracer
from BaselineModels.utils.train_epoch import train, val


print(os.environ.get("CUDA_VISIBLE_DEVICES"))

if torch.cuda.is_available():
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Using CPU.")
print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))
print(torch.cuda.get_device_name(device))



USE_WANDB = True


def main_fct(config = None):
    import argparse

    if USE_WANDB:
        wandb.init(config=config)
        config = wandb.config
        MODEL_CLASS = config.model_class
    else:
        MODEL_CLASS = config['model_class']

    DATASET = 'TUH'    # either 'TUH' or 'BCI' or 'DenoiseNet'
    artifact_type = 'EMG'
    ensemble_loss = False
    torch.manual_seed(0)

    workin_dir = '/system/user/studentwork/gutenber'


    if DATASET == 'BCI':
        raise Exception("BCI dataset TBD")
        config_path = '/content/drive/My Drive/A_EEG/CLEEGN/configs/bci-challenge/config.yml'
        model_config_path = '/content/drive/My Drive/A_EEG/CLEEGN/configs/bci-challenge/model_config.yml'

    elif DATASET == 'TUH':
        config_path = os.path.join(workin_dir, 'configs/config.yml')
        model_config_path = os.path.join(workin_dir, 'configs/model_config.yml')

    elif DATASET == 'DenoiseNet':
        raise Exception("DenoiseNet dataset TBD")
        config_path = '/content/drive/My Drive/A_EEG/CLEEGN/configs/EEG_DenoiseNet/config.yml'
        model_config_path = '/content/drive/My Drive/A_EEG/CLEEGN/configs/EEG_DenoiseNet/model_config.yml'


    model_name = yaml.safe_load(Path(config_path).read_text())['model_name']
    cfg_dataset = yaml.safe_load(Path(config_path).read_text())['Dataset']
    cfg_general = yaml.safe_load(Path(config_path).read_text())
    cfg_model = yaml.safe_load(Path(model_config_path).read_text())[MODEL_CLASS]
    if MODEL_CLASS == 'OneD_Res_CNN':
        window_size = 1.6 #cfg_dataset["window_size"]
    elif MODEL_CLASS == 'IC_U_Net' or MODEL_CLASS == 'CLEEGN':
        window_size = 4
    else:
        window_size = 4
        print('TODO: Unknown model class.')
    stride = window_size/2
    

    if USE_WANDB:
        wandb.config.update(
            {
                "batch_size": cfg_model['batch_size'],
                "normalize": cfg_dataset['normalize'],
                "scheduler":cfg_model['scheduler'],
                "subjects_train": cfg_dataset['subjects_train'],
                "subjects_val": cfg_dataset['subjects_val'],
                "subjects_test": cfg_dataset['subjects_test'],
                "window_size": window_size,
                "stride": stride,
            }
        )

    SFREQ      = cfg_dataset["sfreq"]
    normalize  = cfg_dataset["normalize"]
    NUM_EPOCHS = cfg_general['epochs']
    BATCH_SIZE = cfg_model['batch_size']
    LR         = cfg_model["learning_rate"]
    scheduler_type = cfg_model["scheduler"]['type']


    if scheduler_type == 'MultiStepLR':
        scheduler_milestones = cfg_model["scheduler"]['milestones']
    if scheduler_type == 'MultiStepLR' or scheduler_type == 'ExponentialLR':
        scheduler_gamma = cfg_model["scheduler"]['gamma']


    # Save path
    if cfg_general["save_path"] is None:
        if DATASET == 'DenoiseNet':
            SAVE_PATH = 'logs/' + DATASET + '_' + artifact_type + '/' + MODEL_CLASS
        else:
            SAVE_PATH = 'logs/' + DATASET + '/' + MODEL_CLASS

        if not os.path.exists(SAVE_PATH):
            try:
                os.makedirs(SAVE_PATH)
            except Exception as e:
                print(f"Failed to create directory '{SAVE_PATH}': {e}")
    else:
        SAVE_PATH = cfg_general["save_path"]

    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")

#######################################################################################################
    if DATASET == 'TUH' or DATASET == 'BCI':
        x_train, y_train = create_dataset(
            os.path.join(cfg_dataset["x_basepath"], cfg_dataset["x_fpath"]),
            os.path.join(cfg_dataset["y_basepath"], cfg_dataset["y_fpath"]),
            cfg_dataset["subjects_train"], tmin=cfg_dataset["tmin"], tmax=cfg_dataset["tmax"],
            ch_names=cfg_dataset["ch_names"], win_size=window_size, stride=stride
        )
    elif DATASET == 'DenoiseNet':
        x, y = create_EEG_DenoiseNet_dataset(cfg_dataset, artifact_type, debug = True)
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)
        x_train = np.expand_dims(x_train,1)
        y_train = np.expand_dims(y_train,1)
        x_valid = np.expand_dims(x_valid,1)
        y_valid = np.expand_dims(y_valid,1)

    print(f'x_train shape: {x_train.shape}')



    x_train = np2TT(np.expand_dims(x_train, axis=1))
    y_train = np2TT(np.expand_dims(y_train, axis=1))

    print(f'Max x: {x_train.max()}')
    print(f'Max y: {y_train.max()}')
    
    if MODEL_CLASS == 'xLSTM':
        x_train, y_train = x_train.permute(0,1,3,2).squeeze(), y_train.permute(0,1,3,2).squeeze()

    if DATASET == 'TUH' or DATASET == 'BCI':
        x_valid, y_valid = create_dataset(
            os.path.join(cfg_dataset["x_basepath"], cfg_dataset["x_fpath"]),
            os.path.join(cfg_dataset["y_basepath"], cfg_dataset["y_fpath"]),
            cfg_dataset["subjects_val"], tmin=cfg_dataset["tmin"], tmax=cfg_dataset["tmax"],
            ch_names=cfg_dataset["ch_names"], win_size=window_size, stride=stride
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

    #######################################################################################################

    model = model_select(MODEL_CLASS, cfg_model, cfg_dataset, window_size).to(device)

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

    if scheduler_type == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer = optimizer, milestones = scheduler_milestones, gamma = scheduler_gamma)
    elif scheduler_type == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer, gamma=scheduler_gamma)
    else:
        scheduler = None

    tra_time0 = time.time()
    loss_curve = {"epoch": [], "loss": [], "val_loss": []}

    torch.manual_seed(0)
    for epoch in range(NUM_EPOCHS):
        loss = train(tra_loader, model, criteria, optimizer, MODEL_CLASS, normalize, ensemble_loss, USE_WANDB)
        """ validation """
        val_loss = val(val_loader, model, criteria, MODEL_CLASS, normalize)
        lr = optimizer.param_groups[-1]['lr']
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        print("\rEpoch {}/{} - {:.2f} s - loss: {:.4f} - val_loss: {:.4f} - lr: {:e}".format(
            epoch + 1, NUM_EPOCHS, time.time() - tra_time0, loss, val_loss, lr
        ))
        state = dict(
            epoch=epoch + 1, min_loss=ckpts[0].bound, min_vloss=ckpts[1].bound,
            state_dict=model.state_dict(), loss=loss, val_loss=val_loss, learning_rate=lr
        )
        for ckpt in ckpts:
            improved_flag = ckpt.on_epoch_end(epoch + 1, state)
            if improved_flag:
                best_model_state = copy.deepcopy(model.state_dict())
                print('Best model assigned.')

        if USE_WANDB:
            wandb.log({"epoch": epoch, "loss": loss, "val_loss": val_loss})

        loss_curve["epoch"].append(epoch + 1)
        loss_curve["loss"].append(loss)
        loss_curve["val_loss"].append(val_loss)
    ### End_Of_Train
    savemat(os.path.join(SAVE_PATH, "loss_curve.mat"), loss_curve)


    if DATASET == 'TUH' or DATASET == 'BCI':
        x_test, y_test = create_dataset(
            os.path.join(cfg_dataset["x_basepath"], cfg_dataset["x_fpath"]),
            os.path.join(cfg_dataset["y_basepath"], cfg_dataset["y_fpath"]),
            cfg_dataset["subjects_test"], tmin=cfg_dataset["tmin"], tmax=cfg_dataset["tmax"],
            ch_names=cfg_dataset["ch_names"], win_size=cfg_dataset["window_size"], stride=cfg_dataset["stride"]
        )

    x_test = np2TT(np.expand_dims(x_test, axis=1))
    y_test = np2TT(np.expand_dims(y_test, axis=1))

    if MODEL_CLASS == 'xLSTM':
        x_test, y_test = x_test.permute(0,1,3,2).squeeze(), y_test.permute(0,1,3,2).squeeze()

    testset = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=True
    )

    best_model = model_select(MODEL_CLASS, cfg_model).to(device)
    best_model.load_state_dict(best_model_state)
    test_loss = val(test_loader, best_model, criteria, MODEL_CLASS, normalize)
    print("Test loss on best model: {:.6f}".format(test_loss))
    if USE_WANDB:
        wandb.log({"test_loss": test_loss})



if __name__ == "__main__":
    if USE_WANDB:
        wandb.login()

        sweep_config = {
                'method': 'grid',
            }

        parameters_dict = {
        'model_class': {
            'values': ['CLEEGN', 'IC_U_Net', 'OneD_Res_CNN'] #['LSTM', 'Autoencoder_CNN', 'xLSTM', 'Autoencoder_CNN_LSTM2', 'Autoencoder_CNN_LSTM3', 'Autoencoder_CNN_LSTM4', 'Parallel_CNN_LSTM', 'CLEEGN', 'IC_U_Net', 'OneD_Res_CNN']
            },
        }

        sweep_config['parameters'] = parameters_dict

        sweep_id = wandb.sweep(sweep_config, project="EEG_Denoising")

        wandb.agent(sweep_id, main_fct)
    else:
        config = {
        'model_class': 'Autoencoder_CNN_LSTM3' #['LSTM', 'Autoencoder_CNN', 'xLSTM', 'Autoencoder_CNN_LSTM2', 'Autoencoder_CNN_LSTM3', 'Autoencoder_CNN_LSTM4', 'Parallel_CNN_LSTM', 'CLEEGN', 'IC_U_Net', 'OneD_Res_CNN']
            }
        main_fct(config)