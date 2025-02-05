import lightning as L
import numpy as np
#import plotly.express as px
#import plotly.graph_objects as go
import torch
import torchmetrics
#from plotly.subplots import make_subplots
from torch import optim, nn


#mne.set_config('MNE_BROWSER_BACKEND', 'qt')
#mpl.use('TkAgg')

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
        self.train_ap = torchmetrics.classification.BinaryAveragePrecision()
        self.valid_ap = torchmetrics.classification.BinaryAveragePrecision()
        self.test_ap = torchmetrics.classification.BinaryAveragePrecision()
        self.train_stats = torchmetrics.classification.BinaryStatScores()
        self.valid_stats = torchmetrics.classification.BinaryStatScores()
        self.test_stats = torchmetrics.classification.BinaryStatScores()
        self.pred1Count = 0
        self.pred0Count = 0
        self.step = 0

        self.epoch = 0

        self.t_step = 0
        self.args = args
        self.automatic_optimization = False
        self.test_sum = 0

    def sliding_window(self, inputs, targets, mode='train'):
        losses = []
        loss = 0
        startIdx = 0
        stopIdx = 4 * self.args["sample_rate"]
        main_loss = nn.BCEWithLogitsLoss()
        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        steps = 0
        for i in range(4 * self.args["sample_rate"], inputs.shape[2] + self.args["sample_rate"],
                       self.args["sample_rate"]):
            # split into 4 second windows
            eeg_window = inputs[:, :, startIdx:stopIdx]
            targets_window = targets[:, startIdx:stopIdx]
            #self.analyze_eeg(self.model, [self.model.features[-1]], eeg_window[0], targets_window[0])

            outputs, maps = self.model(eeg_window)
            outputs = outputs.squeeze(1)
            outputs = outputs.type(torch.FloatTensor)
            # calculate length of seizure. If length < sample_rate (1 sec) set labels to zero (no seizure).
            seiz_count = torch.sum(targets_window, 1)
            targets_window[seiz_count < self.args["sample_rate"]] = 0
            targets_window, _ = torch.max(targets_window, 1)
            targets_window = targets_window.type(torch.FloatTensor)
            if mode == 'train':
                opt.zero_grad()
            loss = main_loss(outputs, targets_window)
            if mode == 'train':
                self.manual_backward(loss)
                opt.step()
                #scheduler.step()

            losses.append(loss.item())
            outputs = torch.sigmoid(outputs)
            targets_window = targets_window.type(torch.IntTensor)

            if mode == 'train':
                self.train_auc.update(outputs, targets_window)
                self.train_acc.update(outputs, targets_window)
                self.train_ap.update(outputs, targets_window)
                self.train_stats.update(outputs, targets_window)
            elif mode == 'val':
                self.val_auc.update(outputs, targets_window)
                self.valid_acc.update(outputs, targets_window)
                self.valid_ap.update(outputs, targets_window)
                self.valid_stats.update(outputs, targets_window)
            elif mode == 'test':
                self.test_auc.update(outputs, targets_window)
                self.test_acc.update(outputs, targets_window)
                self.test_ap.update(outputs, targets_window)
                self.test_stats.update(outputs, targets_window)
                self.pred1Count = self.pred1Count + torch.sum(torch.round(outputs))
                self.pred0Count = self.pred0Count + (outputs.shape[0] - torch.sum(torch.round(outputs)))

            # shift window by 1 second
            startIdx = startIdx + self.args["sample_rate"]
            stopIdx = stopIdx + self.args["sample_rate"]
        return np.mean(losses)

    def training_step(self, batch, batch_idx):

        inputs, targets = batch
        loss = self.sliding_window(inputs, targets, 'train')

        self.log("train_loss_step", loss)
        self.log('train_auc_step', self.train_auc.compute())
        self.log('train_acc_step', self.train_acc.compute())
        self.log('train_ap_step', self.train_ap.compute())
        stats = self.train_stats.compute()

        self.log('train_tpr_step', stats[0].item() / (stats[0].item() + stats[3].item()))
        self.log('train_tnr_step', stats[2].item() / (stats[2].item() + stats[1].item()))
        self.train_auc.reset()
        self.train_acc.reset()
        self.train_ap.reset()
        self.train_stats.reset()
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.log(name, param.grad.norm())
        return torch.tensor(loss, requires_grad=False)

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        loss = self.sliding_window(inputs, targets, 'val')
        self.log("val_loss_step", loss)
        self.log('val_auc_step', self.val_auc.compute())
        self.log('val_acc_step', self.valid_acc.compute())
        self.log('val_ap_step', self.valid_ap.compute())
        stats = self.valid_stats.compute()
        if stats[0].item() + stats[3].item() > 0:
            self.log('val_tpr_step', stats[0].item() / (stats[0].item() + stats[3].item()))
        if stats[2].item() + stats[1].item() > 0:
            self.log('val_tnr_step', stats[2].item() / (stats[2].item() + stats[1].item()))
        self.val_auc.reset()
        self.valid_acc.reset()
        self.valid_ap.reset()
        self.valid_stats.reset()
        return torch.tensor(loss, requires_grad=False)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        loss = self.sliding_window(inputs, targets, 'test')
        self.analyze_eeg(self.model, [self.model.layer3[-1]], inputs, targets)
        self.log("test_loss_step", loss)
        self.log('test_auc_step', self.test_auc.compute())
        self.log('test_acc_step', self.test_acc.compute())
        self.log('test_ap_step', self.test_ap.compute())
        stats = self.test_stats.compute()
        if stats[0].item() + stats[3].item() > 0:
            self.log('test_tpr_step', stats[0].item() / (stats[0].item() + stats[3].item()))
        if stats[2].item() + stats[1].item() > 0:
            self.log('test_tnr_step', stats[2].item() / (stats[2].item() + stats[1].item()))
        self.test_sum = self.test_sum + self.test_auc.compute()
        self.t_step = self.t_step + 1
        self.test_auc.reset()
        self.test_acc.reset()
        self.test_ap.reset()
        self.test_stats.reset()
        return torch.tensor(loss, requires_grad=False)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args["lr_init"], weight_decay=1e-6)
        #scheduler = CosineAnnealingWarmUpSingle(optimizer,
        #                                        max_lr=self.args["lr_init"] * math.sqrt(self.args["batch_size"]),
        #                                        epochs=self.args["epochs"],
        #                                        steps_per_epoch=self.args["steps_per_epoch"],
        #                                        div_factor=math.sqrt(self.args["batch_size"]))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        self.log('pred0Count', self.pred0Count)
        self.log('pred1Count', self.pred1Count)
        self.log('auc_avg', self.test_sum / self.t_step)


    '''
    def analyze_eeg(self,model, target_layers, input, targets):
        info = mne.create_info(input.shape[0], sfreq=200, ch_types=['eeg'] * 19)
        raw = mne.io.RawArray(input, info)
        data = raw.get_data()
        times = raw.times
        #plt.plot(data.times, data.get_data().T)
        #plt.xlabel('time (s)')
        #plt.ylabel('MEG data (T)')
        #fig = go.Figure()
        #for idx, channel_data in enumerate(data):
        #    fig.add_trace(go.Scatter(
        #        x=times,
        #        y=channel_data,
        #        mode='lines',
        #        name=raw.info['ch_names'][idx]
        #    ))
        #fig.update_layout(
        #    title="MEG Data Visualization",
        #    xaxis_title="Time (s)",
        #    yaxis_title="MEG Data (T)",
        #    showlegend=True,
        #)

        # Display the plot
        #fig.show()
        n_channels = 10
        step = 1. / n_channels
        fig = make_subplots(
            rows=n_channels, cols=1, shared_xaxes=True,
            vertical_spacing=step / 2  # Adjust spacing between subplots
        )
        ch_names = raw.info['ch_names'][:n_channels]
        # Add each channel as a separate trace
        for ii in range(n_channels):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=data.T[:, ii],
                    mode='lines',
                    name=ch_names[ii]  # Channel name
                ),
                row=ii + 1, col=1  # Specify the row for each trace
            )

        # Add annotations for channel names
        for ii, ch_name in enumerate(ch_names):
            fig.add_annotation(
                x=-0.05, y=0.5,  # Adjust position of annotation
                xref='paper', yref=f'y{ii + 1}',  # Attach to the correct y-axis
                text=ch_name,
                font=dict(size=9),
                showarrow=False
            )
        for ii in range(n_channels):
            fig.update_yaxes(
                showticklabels=False,
                zeroline=False,
                showgrid=False,
                row=ii + 1, col=1
            )
        # Update the layout
        fig.update_layout(
            height=600, width=1000,  # Set figure size
            title="MEG Data Visualization with Shared X-Axis",
            #xaxis_title="Time (s)",
            showlegend=False,  # Disable legend
            margin=dict(l=50, r=50, t=50, b=50)
        )

        # Show the plot
        fig.show()
        # plot target
        #plt.plot(targets.squeeze(0).cpu().numpy())
        targets = targets.cpu().numpy()
        #targets[100:1000] = 1
        fig = px.line(targets)
        fig.show()
        target_class = None
        with GradCAM(model=model, target_layers=target_layers) as gradcam:
            grayscale_cam = gradcam(input_tensor=input.unsqueeze(0).unsqueeze(0), targets=target_class)
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(input.numpy(), grayscale_cam)
            fig = px.imshow(visualization)
            fig.show()

            print(f'GradCAM:')
            #return heatmap, result
'''
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
