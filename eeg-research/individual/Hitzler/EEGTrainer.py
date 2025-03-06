import numpy as np
import torch
import torchmetrics
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from MarginAccuracy import MarginAccuracy
from utils import calculate_best_threshold_stats


class EEGTrainer(nn.Module):

    def __init__(self, model, args: dict, device, trainloader, valloader, writer):
        super().__init__()
        self.model = model
        self.args = args
        self.device = device
        self.val_acc = torchmetrics.classification.Accuracy(task="binary")
        self.val_auc = torchmetrics.classification.BinaryAUROC()
        self.val_ap = torchmetrics.classification.BinaryAveragePrecision()
        self.val_stats = torchmetrics.classification.BinaryStatScores()
        self.val_margin_metric = MarginAccuracy(margin_seconds=5, window_size=4, sample_rate=self.args["sample_rate"])
        self.train_acc = torchmetrics.classification.Accuracy(task="binary")
        self.train_auc = torchmetrics.classification.BinaryAUROC()
        self.train_ap = torchmetrics.classification.BinaryAveragePrecision()
        self.train_stats = torchmetrics.classification.BinaryStatScores()
        self.train_margin_metric = MarginAccuracy(margin_seconds=5, window_size=4, sample_rate=self.args["sample_rate"])
        self.val_step = 0
        self.training_step = 0
        self.test_step = 0
        self.optimizer = optim.AdamW(model.parameters(), lr=args["lr_init"], weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, 15)
        self.main_loss = nn.BCEWithLogitsLoss()
        self.writer = writer
        self.trainloader = trainloader
        self.valloader = valloader

    def validate_sliding_window(self):
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            val_losses = []
            for i, (data, label) in tqdm(enumerate(self.valloader), desc=f"Validating", total=len(self.valloader),
                                         position=0):
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
                    all_preds.append(outputs)
                    targets_window = targets_window.type(torch.IntTensor)
                    all_targets.append(targets_window)
                    self.val_auc.update(outputs, targets_window)
                    self.val_acc.update(outputs, targets_window)
                    self.val_ap.update(outputs, targets_window)

                    start_idx = start_idx + self.args["sample_rate"]
                    stop_idx = stop_idx + self.args["sample_rate"]
        self.writer.add_scalar("val/auc", self.val_auc.compute(), self.val_step)
        self.writer.add_scalar("val/acc", self.val_acc.compute(), self.val_step)
        self.writer.add_scalar("val/ap", self.val_ap.compute(), self.val_step)
        self.writer.add_scalar("val/loss", np.mean(val_losses), self.val_step)
        tpr, tnr, threshold = calculate_best_threshold_stats(torch.cat(all_targets).detach().numpy(),
                                                             torch.cat(all_preds).detach().numpy())
        # use best threshold to calculate margin accuracy
        all_preds = [torch.where(output > threshold, 1, 0) for output in all_preds]
        self.val_margin_metric.update(torch.cat(all_preds), torch.cat(all_targets))
        margin_ons, margin_offs = self.val_margin_metric.compute()
        self.writer.add_scalar("val/margin_acc_onset", margin_ons, self.val_step)
        self.writer.add_scalar("val/margin_acc_offset", margin_offs, self.val_step)
        self.writer.add_scalar("val/tpr", tpr, self.val_step)
        self.writer.add_scalar("val/tnr", tnr, self.val_step)
        self.writer.add_scalar("val/threshold", threshold, self.val_step)
        self.val_auc.reset()
        self.val_acc.reset()
        self.val_ap.reset()
        self.val_stats.reset()
        self.val_margin_metric.reset()
        self.val_step += 1
        return np.mean(val_losses)

    def train_sliding_window(self):
        print("Training started")
        self.model.to(self.device)
        best_val_loss = np.inf
        early_stopping = 0
        for epoch in range(self.args["epochs"]):
            self.model.train()
            train_losses = []
            if early_stopping > 5:
                break
            for i, (data, label) in tqdm(enumerate(self.trainloader), desc=f"Epoch {epoch}",
                                         total=len(self.trainloader), position=0):

                data, label = data.to(self.device), label.to(self.device)
                start_idx = 0
                stop_idx = 4 * self.args["sample_rate"]
                all_preds = []
                all_targets = []
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

                    train_losses.append(loss.item())
                    outputs = torch.sigmoid(outputs)
                    all_preds.append(outputs)
                    targets_window = targets_window.type(torch.IntTensor)
                    all_targets.append(targets_window)

                    self.train_auc.update(outputs, targets_window)
                    self.train_acc.update(outputs, targets_window)
                    self.train_ap.update(outputs, targets_window)

                    # shift window by 1 second
                    start_idx = start_idx + self.args["sample_rate"]
                    stop_idx = stop_idx + self.args["sample_rate"]
                if self.training_step % self.args["log_interval"] == 0:
                    self.writer.add_scalar("train/auc", self.train_auc.compute(), self.training_step)
                    self.writer.add_scalar("train/acc", self.train_acc.compute(), self.training_step)
                    self.writer.add_scalar("train/ap", self.train_ap.compute(), self.training_step)
                    self.writer.add_scalar("train/loss", np.mean(train_losses), self.training_step)
                    self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], self.training_step)
                    tpr, tnr, threshold = calculate_best_threshold_stats(torch.cat(all_targets).detach().numpy(),
                                                                         torch.cat(all_preds).detach().numpy())
                    # use best threshold to calculate margin accuracy
                    all_preds = [torch.where(output >= threshold, 1, 0) for output in all_preds]
                    self.train_margin_metric.update(torch.cat(all_preds), torch.cat(all_targets))
                    margin_ons, margin_offs = self.train_margin_metric.compute()
                    self.writer.add_scalar("train/margin_acc_onset", margin_ons, self.training_step)
                    self.writer.add_scalar("train/margin_acc_offset", margin_offs, self.training_step)
                    self.writer.add_scalar("train/tpr", tpr, self.training_step)
                    self.writer.add_scalar("train/tnr", tnr, self.training_step)
                    self.writer.add_scalar("train/threshold", threshold, self.training_step)
                    self.train_auc.reset()
                    self.train_acc.reset()
                    self.train_ap.reset()
                    self.train_stats.reset()
                    self.train_margin_metric.reset()
                    self.writer.flush()

                self.training_step += 1
                # check every validation interval
                if (i + 1) % round(len(self.trainloader) * self.args["val_interval"]) == 0 or (i + 1) == len(
                        self.trainloader):
                    print("Validating")
                    avg_val_loss = self.validate_sliding_window()
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(self.model.state_dict(),
                                   'best_models/' + self.model.__class__.__name__ + '_best_model.pth')
                        print(f"Model saved to {f'best_model.pth'}")
                        early_stopping = 0
                    else:
                        early_stopping += 1
                    if early_stopping > 5:
                        print("Early stopping")
                        break
                    self.model.train()
            self.scheduler.step()
        print("Training finished")
        return True
