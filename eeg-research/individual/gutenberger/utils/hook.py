import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import from_numpy as np2TT

from os.path import expanduser
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import math
import time
import sys
import os

class Model_Tracer():
    def __init__(self, monitor="loss", mode="min", do_save=False, root=None, prefix="checkpoint"):
        if mode not in ["min", "max"]:
            raise ValueError("mode can only be `min` or `max`")    
        self.mode = mode
        self.monitor = monitor
        self.do_save = do_save
        self.bound = np.inf if mode == "min" else (-np.inf)
        self.root = os.path.join(expanduser("~"), "Downloads") if root is None else root
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs):
        if ((self.mode == "min" and logs[self.monitor] < self.bound) or
            (self.mode == "max" and logs[self.monitor] > self.bound)
        ):
            print("Epoch {}: {} is improved from {:.6f} to {:.6f}".format(
                epoch, self.monitor, self.bound, logs[self.monitor]
            ))
            self.bound = logs[self.monitor]
            if self.do_save:
                filename = "{}.pth".format(self.prefix)
                torch.save(logs, os.path.join(self.root, filename))
                print("Model saved.")
                return True
            else: 
                return False
        return False
    ### End_Of_Class
