import time

import torchmetrics
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from MarginAccuracy import MarginAccuracy
from utils import calculate_best_threshold_stats
from sklearn.metrics import average_precision_score, precision_score, recall_score


class EEGEvaluator(nn.Module):

    def __init__(self, model, args: dict, device, testloader, writer):
        super().__init__()
        self.model = model
        self.args = args
        self.device = device
        self.test_acc = torchmetrics.classification.Accuracy(task="binary")
        self.test_auc = torchmetrics.classification.BinaryAUROC()
        self.test_ap = torchmetrics.classification.BinaryAveragePrecision()
        self.test_stats = torchmetrics.classification.BinaryStatScores()
        self.test_margin_metric = MarginAccuracy(margin_seconds=5, window_size=4, sample_rate=self.args["sample_rate"])

        self.test_step = 0
        self.main_loss = nn.BCEWithLogitsLoss()
        self.pred1_count = 0
        self.pred0_count = 0
        self.writer = writer
        self.testloader = testloader

    def evaluate_sliding_window(self):
        self.model.to(self.device)
        self.model.eval()
        all_preds = []
        all_targets = []
        avg_inference_times = []
        with torch.no_grad():
            test_losses = []
            for i, (data, label) in tqdm(enumerate(self.testloader), desc=f"Evaluating", total=len(self.testloader),
                                         position=0):
                data, label = data.to(self.device), label.to(self.device)
                start_idx = 0
                stop_idx = 4 * self.args["sample_rate"]
                total_time = 0
                for i in range(4 * self.args["sample_rate"], data.shape[2] + self.args["sample_rate"],
                               self.args["sample_rate"]):
                    eeg_window = data[:, :, start_idx:stop_idx]
                    targets_window = label[:, start_idx:stop_idx]
                    start_time = time.perf_counter()
                    outputs, maps = self.model(eeg_window)
                    end_time = time.perf_counter()
                    total_time += end_time - start_time
                    outputs = outputs.squeeze(1)
                    outputs = outputs.type(torch.FloatTensor)
                    seiz_count = torch.sum(targets_window, 1)
                    targets_window[seiz_count < self.args["sample_rate"]] = 0
                    targets_window, _ = torch.max(targets_window, 1)
                    targets_window = targets_window.type(torch.FloatTensor)
                    loss = self.main_loss(outputs, targets_window)
                    test_losses.append(loss.item())
                    outputs = torch.sigmoid(outputs)
                    all_preds.append(outputs)
                    targets_window = targets_window.type(torch.IntTensor)
                    all_targets.append(targets_window)
                    self.test_auc.update(outputs, targets_window)
                    self.test_acc.update(outputs, targets_window)
                    self.test_ap.update(outputs, targets_window)
                    self.test_stats.update(outputs, targets_window)
                    self.pred1_count = self.pred1_count + torch.sum(torch.round(outputs))
                    self.pred0_count = self.pred0_count + (outputs.shape[0] - torch.sum(torch.round(outputs)))
                    start_idx = start_idx + self.args["sample_rate"]
                    stop_idx = stop_idx + self.args["sample_rate"]
                avg_inference_times.append(total_time / 26)

        test_auc = self.test_auc.compute()
        test_acc = self.test_acc.compute()
        test_ap = average_precision_score(torch.cat(all_targets).detach().numpy(),
                                          torch.cat(all_preds).detach().numpy())

        avg_time = np.mean(avg_inference_times)
        self.writer.add_scalar("test/auc", test_auc, self.test_step)
        self.writer.add_scalar("test/acc", test_acc, self.test_step)
        self.writer.add_scalar("test/ap", test_ap, self.test_step)
        self.writer.add_scalar("test/loss", np.mean(test_losses), self.test_step)
        test_tpr, test_tnr, threshold = calculate_best_threshold_stats(torch.cat(all_targets).detach().numpy(),
                                                                       torch.cat(all_preds).detach().numpy())
        # use best threshold to calculate margin accuracy
        all_preds = [torch.where(output >= threshold, 1, 0) for output in all_preds]
        precision = precision_score(torch.cat(all_targets), torch.cat(all_preds))
        recall = recall_score(torch.cat(all_targets), torch.cat(all_preds))
        self.test_margin_metric.update(torch.cat(all_preds), torch.cat(all_targets))
        margin_ons, margin_offs = self.test_margin_metric.compute()
        self.writer.add_scalar("test/margin_acc_onset", margin_ons, self.test_step)
        self.writer.add_scalar("test/margin_acc_offset", margin_offs, self.test_step)

        self.writer.add_scalar("test/tpr", test_tpr, self.test_step)
        self.writer.add_scalar("test/tnr", test_tnr, self.test_step)
        self.writer.add_scalar("test/threshold", threshold, self.test_step)
        self.test_auc.reset()
        self.test_acc.reset()
        self.test_ap.reset()
        return test_auc, test_acc, test_ap, test_tpr, test_tnr, avg_time, margin_ons, margin_offs, precision, recall