import torch
import torch.nn.functional as F
import numpy as np
import time
import wandb
import matplotlib.pyplot as plt

def train(tra_loader, model, criterion, optimizer, model_class, normalize = False, ensemble_loss = False, use_wandb = False, verbose=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()  # switch to train mode
    normalization = 'percentile'  #TODO make variable

    log = ""
    ep_time0 = time.time()
    epoch_loss = np.zeros((len(tra_loader), ))

    torch.manual_seed(0)
    for i, (x_batch, y_batch) in enumerate(tra_loader):
        # print(i, x_batch.shape, y_batch.shape)
        x_batch, y_batch = x_batch.to(device, dtype=torch.float), y_batch.to(device, dtype=torch.float)

        optimizer.zero_grad()

        output = model(x_batch)

        loss = criterion(output, y_batch)

        loss.backward()
        #if use_wandb:
            #log_gradients_to_wandb(model)
        optimizer.step()

        epoch_loss[i] = loss.item()

    return epoch_loss.mean(axis=0)


def val(val_loader, model, criterion, model_class, normalize = False, verbose=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval() # switch to evaluation mode
    normalization = 'percentile'

    log = ""
    ep_time0 = time.time()
    epoch_loss = np.zeros((len(val_loader), ))

    torch.manual_seed(0)
    for i, (x_batch, y_batch) in enumerate(val_loader):
        x_batch, y_batch = x_batch.to(device, dtype=torch.float), y_batch.to(device, dtype=torch.float)

        with torch.no_grad():
            output = model(x_batch)

        loss = criterion(output, y_batch)

        epoch_loss[i] = loss.item()
    return epoch_loss.mean(axis=0)


def log_gradients_to_wandb(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log({f"grad_norms/{name}": param.grad.norm().item()})

def calc_vel_acc_freq(x, y, freq = 128):
    dt = 1.0 / freq  # Time step
    dt = 1.0
    # Velocity
    v_x = torch.diff(x, n=1, dim=-1) / dt
    v_y = torch.diff(y, n=1, dim=-1) / dt

    # Acceleration
    a_x = torch.diff(v_x, n=1, dim=-1) / dt
    a_y = torch.diff(v_y, n=1, dim=-1) / dt

    # Pad the reconstructed velocity and acceleration for consistent size
    v_x = F.pad(v_x, (0, 1), mode='constant')
    v_y = F.pad(v_y, (0, 1), mode='constant')
    a_x = F.pad(a_x, (0, 2), mode='constant')
    a_y = F.pad(a_y, (0, 2), mode='constant')

    # Frequency estimate using Fourier Transform and calculate MSE for frequency spectrum
    fft_x = torch.abs(torch.fft.fft(x, dim=-1))
    fft_y = torch.abs(torch.fft.fft(y, dim=-1))
    return v_x, v_y, a_x, a_y, fft_x, fft_y