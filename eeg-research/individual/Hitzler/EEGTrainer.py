import math

import numpy as np
import torch
import torchmetrics
from torch import nn, optim
from tqdm import tqdm

from individual.Hitzler.cosine_annealing_with_warmupSingle import CosineAnnealingWarmUpSingle


class EEGTrainer(nn.Module):

    def __init__(self, model, args: dict, device, trainloader, valloader, writer):
        super().__init__()
        self.model = model
        self.args = args
        self.device = device
        self.test_acc = torchmetrics.classification.Accuracy(task="binary")
        self.test_auc = torchmetrics.classification.BinaryAUROC()
        self.test_ap = torchmetrics.classification.BinaryAveragePrecision()
        self.test_stats = torchmetrics.classification.BinaryStatScores()
        self.val_acc = torchmetrics.classification.Accuracy(task="binary")
        self.val_auc = torchmetrics.classification.BinaryAUROC()
        self.val_ap = torchmetrics.classification.BinaryAveragePrecision()
        self.val_stats = torchmetrics.classification.BinaryStatScores()
        self.train_acc = torchmetrics.classification.Accuracy(task="binary")
        self.train_auc = torchmetrics.classification.BinaryAUROC()
        self.train_ap = torchmetrics.classification.BinaryAveragePrecision()
        self.train_stats = torchmetrics.classification.BinaryStatScores()
        self.val_step = 0
        self.training_step = 0
        self.test_step = 0
        self.optimizer = optim.Adam(model.parameters(), lr=args["lr_init"], weight_decay=1e-6)

        self.scheduler = CosineAnnealingWarmUpSingle(self.optimizer,
                                                max_lr=args["lr_init"] * math.sqrt(args["batch_size"]),
                                                epochs=args["epochs"],
                                                steps_per_epoch=args["steps_per_epoch"],
                                                div_factor=math.sqrt(args["batch_size"]))
        self.pred1_count = 0
        self.pred0_count = 0
        self.main_loss = nn.BCEWithLogitsLoss()
        self.writer = writer
        self.trainloader = trainloader
        self.valloader = valloader

    def validate_sliding_window(self):

        self.model.eval()
        with torch.no_grad():
            val_losses = []
            for i, (data, label) in tqdm(enumerate(self.valloader), desc=f"Validating", total=len(self.valloader), position=0):
                data, label = data.to(self.device), label.to(self.device)
                start_idx = 0
                stop_idx = 4 * self.args["sample_rate"]
                for i in range(4 * self.args["sample_rate"], data.shape[2] + self.args["sample_rate"],
                               self.args["sample_rate"]):
                    eeg_window = data[:, :, start_idx:stop_idx]
                    targets_window = label[:, start_idx:stop_idx]
                    outputs, maps = self.model(eeg_window)
                    outputs = outputs.squeeze(1)
                    outputs = outputs.type(torch.FloatTensor)
                    seiz_count = torch.sum(targets_window, 1)
                    targets_window[seiz_count < self.args["sample_rate"]] = 0
                    targets_window, _ = torch.max(targets_window, 1)
                    targets_window = targets_window.type(torch.FloatTensor)
                    loss = self.main_loss(outputs, targets_window)
                    val_losses.append(loss.item())
                    outputs = torch.sigmoid(outputs)
                    targets_window = targets_window.type(torch.IntTensor)
                    self.val_auc.update(outputs, targets_window)
                    self.val_acc.update(outputs, targets_window)
                    self.val_ap.update(outputs, targets_window)
                    self.val_stats.update(outputs, targets_window)
                    start_idx = start_idx + self.args["sample_rate"]
                    stop_idx = stop_idx + self.args["sample_rate"]
                if self.val_step % self.args["log_interval"] == 0:
                    self.writer.add_scalar("val/auc", self.val_auc.compute(), self.val_step)
                    self.writer.add_scalar("val/acc", self.val_acc.compute(), self.val_step)
                    self.writer.add_scalar("val/ap", self.val_ap.compute(), self.val_step)
                    self.writer.add_scalar("val/loss", np.mean(val_losses), self.val_step)
                    stats = self.val_stats.compute()
                    if stats[0].item() + stats[3].item() > 0:
                        self.writer.add_scalar('val/tpr', stats[0].item() / (stats[0].item() + stats[3].item()), self.val_step)
                    if stats[2].item() + stats[1].item() > 0:
                        self.writer.add_scalar('val/tnr', stats[2].item() / (stats[2].item() + stats[1].item()), self.val_step)
                    self.val_auc.reset()
                    self.val_acc.reset()
                    self.val_ap.reset()
                    self.val_stats.reset()
                self.val_step += 1
        return np.mean(val_losses)

    def train_sliding_window(self):
        print("Training started")
        print("Model: ", self.model)
        print("Device: ", self.device)
        self.model.to(self.device)
        best_val_loss = np.inf
        early_stopping = 0
        for epoch in range(self.args["epochs"]):
            self.model.train()
            train_losses = []
            if early_stopping > 5:
                break
            for i, (data, label) in tqdm(enumerate(self.trainloader), desc=f"Epoch {epoch}", total=len(self.trainloader), position=0):

                data, label = data.to(self.device), label.to(self.device)
                start_idx = 0
                stop_idx = 4 * self.args["sample_rate"]
                for j in range(4 * self.args["sample_rate"], data.shape[2] + self.args["sample_rate"],
                               self.args["sample_rate"]):
                    # split into 4 second windows
                    eeg_window = data[:, :, start_idx:stop_idx]
                    targets_window = label[:, start_idx:stop_idx]
                    self.optimizer.zero_grad()
                    outputs, maps = self.model(eeg_window)
                    outputs = outputs.squeeze(1)
                    outputs = outputs.type(torch.FloatTensor)
                    # calculate length of seizure. If length < sample_rate (1 sec) set labels to zero (no seizure).
                    seiz_count = torch.sum(targets_window, 1)
                    targets_window[seiz_count < self.args["sample_rate"]] = 0
                    targets_window, _ = torch.max(targets_window, 1)
                    targets_window = targets_window.type(torch.FloatTensor)

                    loss = self.main_loss(outputs, targets_window)
                    loss.backward(loss)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()
                    self.scheduler.step()

                    train_losses.append(loss.item())
                    outputs = torch.sigmoid(outputs)
                    targets_window = targets_window.type(torch.IntTensor)

                    self.train_auc.update(outputs, targets_window)
                    self.train_acc.update(outputs, targets_window)
                    self.train_ap.update(outputs, targets_window)
                    self.train_stats.update(outputs, targets_window)

                    # shift window by 1 second
                    start_idx = start_idx + self.args["sample_rate"]
                    stop_idx = stop_idx + self.args["sample_rate"]

                if self.training_step % self.args["log_interval"] == 0:
                    self.writer.add_scalar("train/auc", self.train_auc.compute(), self.training_step)
                    self.writer.add_scalar("train/acc", self.train_acc.compute(), self.training_step)
                    self.writer.add_scalar("train/ap", self.train_ap.compute(), self.training_step)
                    self.writer.add_scalar("train/loss", np.mean(train_losses), self.training_step)
                    self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], self.training_step)
                    stats = self.train_stats.compute()
                    if stats[0].item() + stats[3].item() > 0:
                        self.writer.add_scalar('train/tpr', stats[0].item() / (stats[0].item() + stats[3].item()), self.training_step)
                    if stats[2].item() + stats[1].item() > 0:
                        self.writer.add_scalar('train/tnr', stats[2].item() / (stats[2].item() + stats[1].item()), self.training_step)
                    self.train_auc.reset()
                    self.train_acc.reset()
                    self.train_ap.reset()
                    self.train_stats.reset()
                    self.writer.flush()

                self.training_step += 1
                # check every validation interval
                if (i + 1) % round(len(self.trainloader) * self.args["val_interval"]) == 0 or (i + 1) == len(self.trainloader):
                    avg_val_loss = self.validate_sliding_window()
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(self.model.state_dict(), 'best_models/'+ self.model.__class__.__name__ + '_best_model.pth')
                        print(f"Model saved to {f'best_model.pth'}")
                        early_stopping = 0
                    else:
                        early_stopping += 1
                    if early_stopping > 5:
                        print("Early stopping")
                        break
                    self.model.train()
        print("Training finished")
        return True
