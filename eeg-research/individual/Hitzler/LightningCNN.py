import lightning as L
import numpy as np
import torch
import torchmetrics
from torch import optim, nn


class LightningCNN(L.LightningModule):

    def __init__(self, model, args: dict):
        super().__init__()
        self.model = model
        self.train_acc = torchmetrics.classification.Accuracy(task="binary")
        self.valid_acc = torchmetrics.classification.Accuracy(task="binary")
        self.test_acc = torchmetrics.classification.Accuracy(task="binary")
        self.train_auc = torchmetrics.classification.BinaryAUROC()
        self.val_auc = torchmetrics.classification.BinaryAUROC()
        self.test_auc = torchmetrics.classification.BinaryAUROC()
        self.train_ap = torchmetrics.AveragePrecision(task="binary")
        self.valid_ap = torchmetrics.AveragePrecision(task="binary")
        self.test_ap = torchmetrics.AveragePrecision(task="binary")
        self.pred1Count = 0
        self.pred0Count = 0
        self.step = 0

        self.epoch = 0

        self.t_step = 0
        self.args = args
        self.automatic_optimization = False

    def sliding_window(self, inputs, targets, mode='train'):
        losses = []
        loss = 0
        startIdx = 0
        stopIdx = 4 * self.args["sample_rate"]
        main_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4]))
        opt = self.optimizers()
        for i in range(0, inputs.shape[2], self.args["sample_rate"]):
            # split into 4 second windows
            eeg_window = inputs[:, :, startIdx:stopIdx]
            targets_window = targets[:, startIdx:stopIdx]
            outputs, maps = self.model(eeg_window)
            outputs = outputs.squeeze(1)
            outputs = outputs.type(torch.FloatTensor)
            # get the max value of the target window (a seizure is present if any value is 1 in the window)
            targets_window, _ = torch.max(targets_window, 1)
            targets_window = targets_window.type(torch.FloatTensor)
            opt.zero_grad()
            loss = main_loss(outputs, targets_window)
            if mode == 'train':
                self.manual_backward(loss)
                opt.step()
            losses.append(loss.item())
            outputs = torch.sigmoid(outputs)
            targets_window = targets_window.type(torch.IntTensor)
            if mode == 'train':
                self.log("train_loss", loss)
                self.train_ap(outputs, targets_window)
                self.log('train_ap_step', self.train_ap)
                self.train_auc(outputs, targets_window)
                self.log('train_auc_step', self.train_auc)
                #outputs = torch.argmax(outputs, dim=1)
                self.train_acc(outputs, targets_window)
                self.log('train_acc_step', self.train_acc)
            elif mode == 'val':
                self.log("val_loss", loss)
                self.valid_ap(outputs, targets_window)
                self.log('valid_ap_step', self.valid_ap)
                self.val_auc(outputs, targets_window)
                self.log('val_auc_step', self.val_auc)
                #outputs = torch.argmax(outputs, dim=1)
                self.valid_acc(outputs, targets_window)
                self.log('val_acc_step', self.valid_acc)
            elif mode == 'test':
                self.log("test_loss", loss)
                self.test_ap(outputs, targets_window)
                self.log('test_ap_step', self.test_ap)
                self.test_auc(outputs, targets_window)
                self.log('test_auc_step', self.test_auc)

                #outputs = torch.argmax(outputs, dim=1)
                self.test_acc(outputs, targets_window)
                self.log('test_acc_step', self.test_acc)
                # get label of preds

                self.pred1Count = self.pred1Count + torch.sum(torch.round(outputs))
                self.pred0Count = self.pred0Count + (outputs.shape[0] - torch.sum(torch.round(outputs)))

            # shift window by 1 second
            startIdx = startIdx + self.args["sample_rate"]
            stopIdx = stopIdx + self.args["sample_rate"]

        return np.mean(losses)

    def training_step(self, batch, batch_idx):

        inputs, targets = batch
        loss = self.sliding_window(inputs, targets, 'train')
        self.step = self.step + 1
        return torch.tensor(loss)

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        loss = self.sliding_window(inputs, targets, 'val')
        self.step = self.step + 1
        return torch.tensor(loss)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        loss = self.sliding_window(inputs, targets, 'test')
        self.step = self.step + 1
        return torch.tensor(loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-6)
        return optimizer

    def on_train_epoch_end(self):
        # log epoch metric
        self.log('train_acc_epoch', self.train_acc)
        self.step = 0
        self.epoch = self.epoch + 1

    def on_validation_epoch_end(self) -> None:
        self.log('val_acc_epoch', self.valid_acc)
        self.log('val_auc', self.val_auc)

    def on_test_epoch_end(self) -> None:
        self.t_step = 0
        self.log('test_acc_epoch', self.test_acc)
        self.log('pred0Count', self.pred0Count)
        self.log('pred1Count', self.pred1Count)

    '''
    def plot_with_color_gradient(self, sequence, importance):
        # transpose sequence to channels x sequence length
        sequence = sequence.T
        importance = importance.T
        num_channels, seq_length = sequence.shape
        fig, axs = plt.subplots(num_channels, 1, figsize=(10, num_channels * 2), constrained_layout=True)

        # Normalize importance for color mapping across all channels
        norm = Normalize(vmin=importance.min(), vmax=importance.max())
        cmap = plt.get_cmap('viridis')

        for i in range(num_channels):
            x = np.arange(seq_length + 1)
            y = np.append(sequence[i], 0)
            c = np.append(importance[i], 0)

            # Create a set of line segments so that we can color them individually
            # This creates the points as a N x 1 x 2 array so that we can stack points
            # together easily to get the segments. The segments array ends up being N x 2 x 2
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create a LineCollection from the segments
            lc = LineCollection(segments, cmap=cmap, norm=norm)

            # Set the values used for colormapping
            lc.set_array(c)
            lc.set_linewidth(2)

            # Plot the lines on the current subplot
            line = axs[i].add_collection(lc)
            axs[i].set_xlim(x.min(), x.max())
            axs[i].set_ylim(y.min(), y.max())
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            # Remove y-axis labels for clarity if desired
            # axs[i].set_ylabel(f'Channel {i + 1}')

        # Adding color bar
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axs, orientation='vertical', label='Importance')
        return fig

    def gradcam_analysis(self, batch):
        tensorboard = self.logger.experiment
        # we will save the conv layer weights in this list
        model_weights = []
        conv_layers = []
        # get all the model children as list
        model_children = list(self.model.children())
        # counter to keep count of the conv layers
        counter = 0
        # append all the conv layers and their respective wights to the list
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv1d:
                counter += 1
                model_weights.append(model_children[i].weight)
                conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == nn.Conv1d:
                            counter += 1
                            model_weights.append(child.weight)
                            conv_layers.append(child)
        cam = EigenCAM(model=self.cnn_wrap_model, target_layers=[conv_layers[-1]])
        eeg_cam = cam(input_tensor=batch[0])[0, :]
        # channel in range(eeg_cam.shape[1]):
        tensorboard.add_figure("Gradcam",
                               self.plot_with_color_gradient(batch[0][0, :, :].cpu().numpy(), eeg_cam),
                               global_step=self.t_step)
    def analyze_edf(self, edf, batch):
        tensorboard = self.logger.experiment
        epoch = mne.read_epochs(edf, preload=False)
        tensorboard.add_figure('eeg', epoch.plot(show_scrollbars=False, show=False), global_step=self.t_step)
        # we will save the conv layer weights in this list
        model_weights = []
        # we will save the 49 conv layers in this list
        conv_layers = []
        # get all the model children as list
        model_children = list(self.model.children())
        # counter to keep count of the conv layers
        counter = 0
        # append all the conv layers and their respective wights to the list
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv1d:
                counter += 1
                model_weights.append(model_children[i].weight)
                conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == nn.Conv1d:
                            counter += 1
                            model_weights.append(child.weight)
                            conv_layers.append(child)
        print(f"Total convolution layers: {counter}")
        outputs = []
        names = []
        edf_data = torch.tensor(torch.tensor(epoch.get_data(), dtype=torch.float)).to(device='cuda:0')
        for layer in conv_layers[0:]:
            edf_data = layer(edf_data)
            outputs.append(edf_data)
            names.append(str(layer))
        processed = []
        for feature_map in outputs:
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map, 0)
            gray_scale = gray_scale / feature_map.shape[0]
            processed.append(gray_scale.data.cpu().numpy())
        fig = plt.figure(figsize=(20, 10))

        for i in range(len(processed)):
            a = fig.add_subplot(5, 4, i + 1)
            plt.plot(processed[i])

            a.set_title(names[i].split('(')[0] + " " + str(i), fontsize=30)
        tensorboard.add_figure('feature_maps', fig, global_step=self.t_step)

        fig = plt.figure(figsize=(20, 10))
        for i in range(len(processed)):
            a = fig.add_subplot(5, 4, i + 1)
            plt.plot(processed[i])
            a.set_title(names[i].split('(')[0] + " " + str(i), fontsize=10)
        a = fig.add_subplot(5, 4, len(processed) + 1)
        plt.plot(epoch.get_data().squeeze(0).mean(axis=0))
        a.set_title("EEG", fontsize=10)
        tensorboard.add_figure('overview', fig, global_step=self.t_step)
        # get overlay plot
        fig = plt.figure(figsize=(20, 10))
        for i in range(len(processed)):
            a = fig.add_subplot(5, 4, i + 1)
            plt.plot(processed[i])
            plt.plot(epoch.get_data().squeeze(0).mean(axis=0))
            a.set_title(names[i].split('(')[0] + " " + str(i), fontsize=10)

        tensorboard.add_figure('overlay', fig, global_step=self.t_step)
        d = dtw.distance_fast(epoch.get_data().squeeze(0).mean(axis=0).astype(np.double),
                              outputs[-1].squeeze(0).mean(axis=0).cpu().detach().numpy().astype(np.double))
        self.log('dtw', d)
        
        titles = ['Approximation', ' Horizontal detail',
                  'Vertical detail', 'Diagonal detail']
        coeffs2 = pywt.dwt(epoch.get_data().squeeze(0), 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        fig = plt.figure(figsize=(12, 3))
        for i, a in enumerate([LL, LH, HL, HH]):
            ax = fig.add_subplot(1, 4, i + 1)
            ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
            ax.set_title(titles[i], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        tensorboard.add_figure('wavelet'+str(self.epoch), fig, global_step = self.step)
        
        # get the GradCAM
        self.gradcam_analysis(batch)
'''
