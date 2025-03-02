import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchinfo import summary

import numpy as np
import math
import json
import time
import mne
import sys
import os
import wandb
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def val(val_loader, model, criterion, model_class, normalize = False, verbose=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval() # switch to evaluation mode
    normalization = 'zero_one'

    log = ""
    ep_time0 = time.time()
    epoch_loss = np.zeros((len(val_loader), ))
    for i, (x_batch, y_batch) in enumerate(val_loader):
        x_batch, y_batch = x_batch.to(device, dtype=torch.float), y_batch.to(device, dtype=torch.float)

        if normalize:
            #TODO: either restore (output times std) and calc loss on restored, or calc loss on normalized and only restore at inference
            # Calculate the standard deviation for each batch along channel and sequence dimensions
            x_batch_reshaped = x_batch.permute(2,0,1,3).flatten(start_dim=1, end_dim=3)
            percentile_95_per_batch = torch.quantile(torch.abs(x_batch_reshaped), 0.95, dim=1)
            if normalization == 'zero_one':
                for channel in range(x_batch.shape[2]):
                    x_min = torch.min(x_batch[:,:,channel,:])
                    x_max = torch.max(x_batch[:,:,channel,:])
                    x_batch[:,:,channel,:] = (x_batch[:,:,channel,:]-x_min)/(x_max-x_min)
                    y_batch[:,:,channel,:] = (y_batch[:,:,channel,:]-x_min)/(x_max-x_min)
            elif normalization == 'percentile':
                for ch in range(percentile_95_per_batch.shape[0]):
                    x_batch[:,:,ch,:] = x_batch[:,:,ch,:]/percentile_95_per_batch[ch] 
                    y_batch[:,:,ch,:] = y_batch[:,:,ch,:]/percentile_95_per_batch[ch] 

        with torch.no_grad():
            if model_class == "Seq2Seq" or model_class == "Seq2Seq_Attention" or model_class == 'Transformer':
                output = model(x_batch, y_batch)
            elif model_class =='CLEEGN2Vec':
                output, _ = model(x_batch)  #only recosntruction loss in val()
            else:
                output = model(x_batch)

        #if normalize:
            #for j in range(std_per_batch.shape[0]):
                #output[j] = output[j]*std_per_batch[j] 

        # output = x_batch # DEBUG
        loss = criterion(output, y_batch)

        epoch_loss[i] = loss.item()
        if verbose:
            print("\r{}".format(" " * len(log)), end="")
            log = "\r{}/{} - {:.4f} s - loss: {:.4f} - acc: nan".format(
                i + 1, len(val_loader), time.time() - ep_time0, epoch_loss[i]
            )
            print(log, end="")
    return epoch_loss.mean(axis=0)

def log_gradients_to_wandb(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log({f"grad_norms/{name}": param.grad.norm().item()})

def calc_vel_acc_freq(x, y, freq = 128):
    dt = 1.0 / freq  # Time step
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


def latent_prediction_loss(z_predicted, z_target, loss_type="mse", alpha=0.5):
    if loss_type == "mse":
        return F.mse_loss(z_predicted, z_target)
    elif loss_type == "cosine":
        cosine_sim = F.cosine_similarity(z_predicted, z_target, dim=-1)
        return 1 - cosine_sim.mean()
    elif loss_type == "combined":
        mse_loss = F.mse_loss(z_predicted, z_target)
        cosine_sim = F.cosine_similarity(z_predicted, z_target, dim=-1)
        return alpha * mse_loss + (1 - alpha) * (1 - cosine_sim.mean())
    else:
        raise ValueError("Unsupported loss type.")

def train(tra_loader, model, criterion, optimizer, model_class, normalize = False, ensemble_loss = False, use_wandb = False, verbose=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()  # switch to train mode
    normalization = 'zero_one'  #TODO make variable

    log = ""
    ep_time0 = time.time()
    epoch_loss = np.zeros((len(tra_loader), ))
    for i, (x_batch, y_batch) in enumerate(tra_loader):
        # print(i, x_batch.shape, y_batch.shape)
        x_batch, y_batch = x_batch.to(device, dtype=torch.float), y_batch.to(device, dtype=torch.float)

        optimizer.zero_grad()


        if normalize:
            x_batch_reshaped = x_batch.permute(2,0,1,3).flatten(start_dim=1, end_dim=3)
            percentile_95_per_batch = torch.quantile(torch.abs(x_batch_reshaped), 0.95, dim=1)
            if normalization == 'zero_one':
                for channel in range(x_batch.shape[2]):
                    x_min = torch.min(x_batch[:,:,channel,:])
                    x_max = torch.max(x_batch[:,:,channel,:])
                    x_batch[:,:,channel,:] = (x_batch[:,:,channel,:]-x_min)/(x_max-x_min)
                    y_batch[:,:,channel,:] = (y_batch[:,:,channel,:]-x_min)/(x_max-x_min)
            elif normalization == 'percentile':
                for ch in range(percentile_95_per_batch.shape[0]):
                    x_batch[:,:,ch,:] = x_batch[:,:,ch,:]/percentile_95_per_batch[ch] 
                    y_batch[:,:,ch,:] = y_batch[:,:,ch,:]/percentile_95_per_batch[ch] 

        if model_class == "Seq2Seq" or model_class == 'Transformer' or model_class == "Seq2Seq_Attention":
            output = model(x_batch, y_batch)
        elif model_class =='CLEEGN2Vec':
            reconstructed_eeg, latent_noisy = model(x_batch)
            latent_clean = model.encode(y_batch)
        else:
            output = model(x_batch)

        #if normalize:
            #for j in range(std_per_batch.shape[0]):
                #output[j] = output[j]*std_per_batch[j] 

        if ensemble_loss:
            v_x, v_y, a_x, a_y, fft_x, fft_y = calc_vel_acc_freq(output, y_batch)
            loss_ampl = criterion(output, y_batch)
            loss_vel = criterion(v_x, v_y)
            loss_acc = criterion(a_x, a_y)
            loss_freq = criterion(fft_x, fft_y)
            loss = loss_ampl + loss_vel + loss_acc + loss_freq
        elif model_class == 'CLEEGN2Vec':
            reconstruction_loss = criterion(reconstructed_eeg, y_batch)
            latent_loss = latent_prediction_loss(latent_noisy, latent_clean, loss_type="mse")
            # Combine Losses
            loss = reconstruction_loss + 0.1 * latent_loss
        else:
            loss = criterion(output, y_batch)

        loss.backward()
        #if use_wandb:
            #log_gradients_to_wandb(model)
        optimizer.step()

        epoch_loss[i] = loss.item()

        if (np.isnan(loss.item())):
            print(x_batch)
            print(x_batch)
        if verbose:
            print("\r{}".format(" " * len(log)), end="")
            log = "\r{}/{} - {:.4f} s - loss: {:.4f} - acc: nan".format(
                i + 1, len(tra_loader), time.time() - ep_time0, epoch_loss[i]
            )
            print(log, end="")
            
            #if (i == 0 or i == len(tra_loader)-1 or i%10 == 0):
            if (0==1):        
                x_b_nump = x_batch.numpy()
                y_b_nump = y_batch.numpy()
                out = output.detach().numpy()
                
                plt.plot(x_b_nump[0,:,:][0,:], label = 'x')
                plt.plot(y_b_nump[0,:,:][0,:], label = 'y')
                plt.plot(out[0,:,:][0,:], label = 'out')
                plt.legend()
                plt.savefig("test.pdf", format="pdf", bbox_inches="tight")
                plt.show()

                # CLEEGN:
                #plt.plot(x_b_nump[0,:,1,:][0,:])
                #plt.plot(y_b_nump[0,:,1,:][0,:])
                #plt.plot(out[0,:,1,:][0,:])
                #plt.show()

            
                # mse = np.zeros(18)
                # for i in range(18):
                #     plt.plot(x_b_nump[0,:,i,:][0,:])
                #     plt.plot(y_b_nump[0,:,i,:][0,:])
                #     plt.plot(out[0,:,i,:][0,:])
                #     plt.show()
                #     mse[i] = np.mean((y_b_nump[0,:,i,:][0,:]-out[0,:,i,:][0,:])**2)
                
    return epoch_loss.mean(axis=0)
