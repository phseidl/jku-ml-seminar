import glob
import os

import mne
import plotly.express as px
import plotly.graph_objects as go
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from plotly.subplots import make_subplots
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from individual.Hitzler.utils import read_json
from models.eegnet_pytorch import EEGNet

from pytorch_grad_cam  import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.preprocessing import normalize

data = (glob.glob(os.path.join("../data/processed/dev/edf", "*.fif")))
labels = (glob.glob(os.path.join("../data/processed/dev/labels", "*labels.npy")))

MODEL_DICT = {
    "EEGNet": EEGNet

}

if __name__ == "__main__":
    config = read_json('practical work/configs/config.json')
    test_config = read_json('practical work/configs/test.json')
    model = MODEL_DICT[test_config["model"]](config, torch.device(config["device"]))

    model.load_state_dict(torch.load(test_config["model_path"]))
    target_layers = [model.block2[0]]


    input = torch.tensor(normalize(np.squeeze(mne.read_epochs(data[0], verbose=40).get_data(copy=True)).astype(np.float32))).unsqueeze(0)

    targets = torch.tensor(np.load(labels[0]).astype(np.float32))
    info = mne.create_info(input.shape[1], sfreq=200, ch_types=['eeg'] * 19)
    raw = mne.io.RawArray(input[0], info)
    data = raw.get_data()
    times = raw.times
    # plt.plot(data.times, data.get_data().T)
    # plt.xlabel('time (s)')
    # plt.ylabel('MEG data (T)')
    # fig = go.Figure()
    # for idx, channel_data in enumerate(data):
    #    fig.add_trace(go.Scatter(
    #        x=times,
    #        y=channel_data,
    #        mode='lines',
    #        name=raw.info['ch_names'][idx]
    #    ))
    # fig.update_layout(
    #    title="MEG Data Visualization",
    #    xaxis_title="Time (s)",
    #    yaxis_title="MEG Data (T)",
    #    showlegend=True,
    # )

    # Display the plot
    # fig.show()
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
        # xaxis_title="Time (s)",
        showlegend=False,  # Disable legend
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Show the plot
    fig.show()
    # plot target
    # plt.plot(targets.squeeze(0).cpu().numpy())
    targets = targets.cpu().numpy()
    # targets[100:1000] = 1
    fig = px.line(targets)
    fig.show()
    target_class = None
    input = input.unsqueeze(0)

    #targets = targets[np.newaxis, :]
    target_class = [BinaryClassifierOutputTarget(1)]
    for i in range(0, input.shape[-1], 200):
        input_window = input[:, :, :, i:i + 800]
        #target_class = targets[i:i + 800]
        with GradCAM(model=model, target_layers=target_layers) as gradcam:
            grayscale_cam = gradcam(input_tensor=input_window, targets=target_class)
            grayscale_cam = grayscale_cam[0][0]
            #visualization = show_cam_on_image(input.numpy(), grayscale_cam)
            #fig = px.imshow(visualization)
            #fig.show()
            # Normalize EEG data for overlay
            eeg_normalized = input_window
            eeg_signal = eeg_normalized[0][0][0]
            time = np.linspace(0, 4, len(eeg_signal))


            norm = mcolors.Normalize(vmin=np.min(grayscale_cam), vmax=np.max(grayscale_cam))
            cmap = cm.get_cmap("jet")
            points = np.array([time, eeg_signal.numpy()]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            colors = cmap(norm(grayscale_cam[:-1]))

            fig, ax = plt.subplots(figsize=(12, 5))
            line = LineCollection(segments, colors=colors, linewidth=2)
            ax.add_collection(line)
            ax.set_xlim(time.min(), time.max())
            ax.set_ylim(eeg_signal.min(), eeg_signal.max())

            # Add colorbar for Grad-CAM activation
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Grad-CAM Activation")

            # Labels and title
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("EEG Signal Amplitude")
            ax.set_title("EEG Signal Colored by Grad-CAM Activation")

            plt.show()

            #grayscale_cam = (grayscale_cam - np.min(grayscale_cam)) / (np.max(grayscale_cam) - np.min(grayscale_cam))

            # Plot the EEG signal with the Grad-CAM heatmap
            #plt.figure(figsize=(12, 5))
            #plt.xticks(np.arange(0, max(np.arange(800) / 200) + 1, 4))
            #for i in range(19):
            #    plt.plot(eeg_normalized[0][i] + i * 0.5, label=f"EEG Channel {i}")
            #plt.plot(eeg_normalized[0][0][0], label="EEG Signal")

            #plt.imshow(grayscale_cam, aspect="auto", cmap="jet", alpha=0.5,
            #           extent=[0, 4, np.min(eeg_normalized), np.max(eeg_normalized) + 90])
            #plt.plot(grayscale_cam, label="Grad-CAM Heatmap", alpha=0.5)
            #plt.title("Grad-CAM Heatmap for EEG Seizure Detection")
            #plt.xlabel("Time")
            #plt.ylabel("EEG Signal Amplitude")
            #plt.xlim(0, 4)
            #plt.xticks(np.arange(0, 5, 1))
            #plt.grid(True, linestyle="--", alpha=0.5)
            #plt.legend()
            #plt.show()

            print(f'GradCAM:')
            # return heatmap, result