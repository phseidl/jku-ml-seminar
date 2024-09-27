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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def val(val_loader, model, criterion, verbose=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval() # switch to evaluation mode

    log = ""
    ep_time0 = time.time()
    epoch_loss = np.zeros((len(val_loader), ))
    for i, (x_batch, y_batch) in enumerate(val_loader):
        x_batch, y_batch = x_batch.to(device, dtype=torch.float), y_batch.to(device, dtype=torch.float)
        with torch.no_grad():
            output = model(x_batch)
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


def train(tra_loader, model, criterion, optimizer, verbose=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()  # switch to train mode

    log = ""
    ep_time0 = time.time()
    epoch_loss = np.zeros((len(tra_loader), ))
    for i, (x_batch, y_batch) in enumerate(tra_loader):
        # print(i, x_batch.shape, y_batch.shape)
        x_batch, y_batch = x_batch.to(device, dtype=torch.float), y_batch.to(device, dtype=torch.float)

        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
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
