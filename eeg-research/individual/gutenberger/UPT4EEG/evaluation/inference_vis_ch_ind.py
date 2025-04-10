import torch
if torch.cuda.is_available():
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Using CPU.")
print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))
#print(torch.cuda.get_device_name(device))

import os
print(os.environ.get("CUDA_VISIBLE_DEVICES"))

from torch import from_numpy as np2TT
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
from pathlib import Path

from UPT4EEG.utils.build import create_dataset
from UPT4EEG.utils.misc import group_data_per_chan
from UPT4EEG.dataset.sparse_eeg_dataset_val2 import SparseEEGDataset_val
from UPT4EEG.dataset.collator_val import SparseEEG_Collator
from UPT4EEG_mp_2.model.encoder import Encoder
from UPT4EEG_mp_2.model.decoder import DecoderPerceiver
from UPT4EEG_mp_2.model.UPT4EEG import UPT4EEG






DATASET = 'TUH'    # either 'TUH' or 'BCI' or 'DenoiseNet'
MODEL_CLASS = 'UPT4EEG'
saved_model_type = 'big_rand' #'small_tuh' 

config_path = '/system/user/studentwork/gutenber/configs/config.yml'
#model_config_path = '/content/drive/My Drive/A_EEG/CLEEGN/configs/tusz/model_config.yml'

model_name = yaml.safe_load(Path(config_path).read_text())['model_name']
cfg_dataset = yaml.safe_load(Path(config_path).read_text())['Dataset']
cfg_general = yaml.safe_load(Path(config_path).read_text())
#cfg_model = yaml.safe_load(Path(model_config_path).read_text())[MODEL_CLASS]


SFREQ      = cfg_dataset["sfreq"]
normalize  = cfg_dataset["normalize"]
#use_montage = cfg_dataset['use_montage']

#use_montage = 'tuh'
#NUM_EPOCHS = 1 #cfg_general['epochs']
#BATCH_SIZE = cfg_model['batch_size']
#LR         = cfg_model["learning_rate"]


SAVE_PATH = 'logs/' + DATASET + '/' + MODEL_CLASS

if not os.path.exists(SAVE_PATH):
    try:
        os.makedirs(SAVE_PATH)
    except Exception as e:
        print(f"Failed to create directory '{SAVE_PATH}': {e}")


timestamp = datetime.now().strftime("%b%d_%H-%M-%S")

if saved_model_type == 'big_rand':
    model_weights_name = 'Jan18_10-47-07_val' #big model on random, iosame=False
    d_model = 192*4
    dim = 256   # ~6M parameter model
    num_heads = 8 #3
    depth = 6
#model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan19_17-49-57_val.pth'  #trained with bin loss!!
elif saved_model_type == 'small_tuh':
    model_weights_name = 'Jan19_18-47-03_val' #small model on tuh
    d_model = 192*2
    dim = 192   # ~6M parameter model
    num_heads = 4 #3
    depth = 3

model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_' + model_weights_name + '.pth'  #small model on tuh

# hyperparameters
num_supernodes = 512
input_dim = 1
output_dim = 1
use_mlp_posEnc = True


# initialize model
model = UPT4EEG(
    encoder = Encoder(
        input_dim=input_dim,
        # for EEG, we use ndim just for time pos encoding, time is 1D --> ndim = 1
        ndim=1,
        # d_model
        gnn_dim=d_model,
        # ViT-T latent dimension
        enc_dim=dim,
        enc_num_heads=num_heads,
        # ViT-T has 12 blocks -> parameters are split evenly among encoder/approximator/decoder
        enc_depth=depth,
        # the perceiver is optional, it changes the size of the latent space to NUM_LATENT_TOKENS tokens
        # perc_dim=dim,
        # perc_num_heads=num_heads,
        # num_latent_tokens=32,
        mlp_pos_enc=use_mlp_posEnc,
    ),
    decoder=DecoderPerceiver(
        # tell the decoder the dimension of the input (dim of approximator)
        input_dim=dim,
        output_dim=output_dim,
        # images have 2D coordinates
        ndim=1,
        # as in ViT-T
        dim=dim,
        num_heads=num_heads,
        # ViT-T has 12 blocks -> parameters are split evenly among encoder/approximator/decoder
        depth=depth,
        mlp_pos_enc=use_mlp_posEnc,
    ),
)

model = model.to(device)
print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")


state_path = os.path.join(model_path)
state = torch.load(state_path, map_location="cpu")
model.load_state_dict(state["state_dict"])


window_size = 1
stride = 0.5

subject_vis = ["016"]

#cfg_dataset["subjects_test"]

x_test, y_test, ch_names = create_dataset(
    os.path.join(cfg_dataset["x_basepath"], cfg_dataset["x_fpath"]),
    os.path.join(cfg_dataset["y_basepath"], cfg_dataset["y_fpath"]),
    [subject_vis], tmin=506., tmax=cfg_dataset["tmax"],
    ch_names=cfg_dataset["ch_names"], win_size=window_size, stride=stride
)

#cfg_dataset["window_size"]
#x_train.shape: [Nr of segments, channel nr, sequence length]
x_test = np2TT(x_test)
y_test = np2TT(y_test)


io_same = True
use_montage = 'random' #'user_specific'

query_montage_pairs = [
            ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
            ]
query_montage_pairs = [
            ('FP1', 'F7'), 
            ('F7', 'T3'), 
            ('T3', 'T5'), 
            ('T5', 'O1'),
            ('FP2', 'F8'), 
            ('F8', 'T4'), 
            ('T4', 'T6'), 
            ('T6', 'O2'),
            ('A1', 'T3'), 
            ('T3', 'C3'), 
            ('C3', 'CZ'), 
            ('CZ', 'C4'),
            ('C4', 'T4'), 
            ('T4', 'A2'), 
            ('FP1', 'F3'), 
            ('F3', 'C3'),
            ('C3', 'P3'), 
            ('P3', 'O1'),
            ('C4', 'P4'), 
            ('P4', 'O2')
            ]

inp_montage_pairs = [
            ('FP1', 'F7'), 
            ('F7', 'T3'), 
            ('T3', 'T5'), 
            ('T5', 'O1'),
            ('FP2', 'F8'), 
            ('F8', 'T4'), 
            ('T4', 'T6'), 
            ('T6', 'O2'),
            ('A1', 'T3'), 
            ('T3', 'C3'), 
            ('C3', 'CZ'), 
            ('CZ', 'C4'),
            ('C4', 'T4'), 
            ('T4', 'A2'), 
            ('FP1', 'F3'), 
            ('F3', 'C3'),
            ('C3', 'P3'), 
            ('P3', 'O1'),
            ('C4', 'P4'), 
            ('P4', 'O2')
            ]




query1 = {'query_freq': 500.0, 'query_montage_pairs': query_montage_pairs}
query2 = {'query_freq': 250.0, 'query_montage_pairs': query_montage_pairs}
query3 = {'query_freq': 500.0, 'query_montage_pairs': query_montage_pairs}
query4 = {'query_freq': 3000.0, 'query_montage_pairs': query_montage_pairs}

query_list = [query1, query2, query3, query4]

for query in query_list:
    test_dataset = SparseEEGDataset_val(x_test, 
                                        y_test, 
                                        query, 
                                        ch_names, 
                                        cfg_dataset, 
                                        use_montage=use_montage, 
                                        montage_pairs_input=inp_montage_pairs,
                                        n_input_chs=len(inp_montage_pairs),
                                        n_output_chs=len(query['query_montage_pairs']),
                                        io_same=io_same)

    sample = test_dataset[0]
    print(f"Input features shape: {sample['input_feat'].shape}")
    print(f"Input positions shape: {sample['input_pos'].shape}")
    print(f"Target features shape: {sample['target_feat'].shape}")
    print(f"Output positions shape: {sample['target_pos'].shape}")
    print(f"Query positions shape: {sample['query_pos'].shape}")

    # setup dataloader
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        collate_fn=SparseEEG_Collator(num_supernodes=num_supernodes, deterministic=False),
    )

    test_batch = next(iter(test_dataloader))

    # make predictions
    with torch.no_grad():
        y_hat = model(
            input_feat=test_batch["input_feat"].to(device),
            input_pos=test_batch["input_pos"].to(device),
            batch_idx=test_batch["batch_idx"].to(device),
            output_pos=test_batch["query_pos"].to(device),
        ).squeeze().cpu()

    x_noisy = test_batch["input_feat"] #noisy input signal
    y_tar = test_batch["target_feat"] #clean target signal


    t_tar = test_batch["target_pos"].squeeze()[:, -1]
    t = test_batch["query_pos"].squeeze()[:, -1]
    t_x = test_batch["input_pos"].squeeze()[:, -1]

    output_pos  = np.array(test_batch["target_pos"].squeeze()[:, 0:6])
    input_pos  = np.array(test_batch["input_pos"].squeeze()[:, 0:6])
    query_pos  = np.array(test_batch["query_pos"].squeeze()[:, 0:6])


    channel_data, channel_time = group_data_per_chan(query_pos, y_hat, t)
    channel_data_tar, channel_time_tar = group_data_per_chan(output_pos, y_tar, t_tar)
    channel_data_x, channel_time_x = group_data_per_chan(input_pos, x_noisy, t_x)

    print(f'y_hat shape: {y_hat.shape}')
    print(f'y_tar shape: {y_tar.shape}')
    print(f'x shape: {x_noisy.shape}')
    print(f'output pos shape: {test_batch["target_pos"].shape}')
    print(f't shape: {t.shape}')

    #from matplotlib.cm import get_cmap
    #cmap = get_cmap("tab20")  # Use a colormap, e.g., 'tab10', 'viridis', etc.
    #colors = cmap(np.linspace(0, 1, len(channel_data)))  # Get distinct colors


    offset = 0  # Starting offset
    idx = 0

    for (channel_pos, data_pred), (channel_pos_tar, data_tar) in zip(channel_data.items(), channel_data_tar.items()):
        try:
            data_x = channel_data_x[channel_pos]
        except:
            data_x = np.zeros_like(data_tar)
        
        idx += 1
        if idx < 111111:
            plt.figure(figsize=(12, 6))
            data_pred = np.array(data_pred)
            data_tar = np.array(data_tar)
            t = np.array(channel_time[channel_pos])
            t_tar = np.array(channel_time_tar[channel_pos_tar])
            if query["query_freq"] < 10:
                plt.scatter(t/173, data_pred, label=f'CLEAN', color="blue", alpha = 0.8)
            else:
                plt.plot(t/173, data_pred, label=f'CLEAN (' + str(int(query["query_freq"])) + ' Hz)', color="blue", linewidth=3, alpha = 0.8)
            plt.plot(t_tar/173, data_tar, label=f'Clean (' + str(int(SFREQ)) + ' Hz)', color='red', linewidth=3, alpha = 0.8)
            plt.plot(t_tar/173, data_x, label=f'Noisy (' + str(int(SFREQ)) + ' Hz)', color='gray', linewidth=3, alpha = 0.5)
            offset += np.max(np.abs(data_pred)) * 1.2  # Increase the offset for the next channel
            

            # Add labels and title
            plt.xlabel('Time (s)', fontsize = 20)
            plt.ylabel('C3-Cz (norm. amp.)', fontsize = 20)
            plt.legend(loc='upper right', fontsize = 20)
            plt.title('Query freq: ' + str(int(query["query_freq"]))  + ' Hz', fontsize = 25)
            plt.xlim((t[0]/SFREQ, 1.0))
            plt.xticks(fontsize = 15)
            plt.yticks(fontsize = 15)
            
            # Show the plot
            plt.tight_layout()
        
            plt.savefig("UPT4EEG/evaluation/plots/ch_independence_plot_S" + subject_vis[0] + "_" + str(query["query_freq"]) + "_" + str(idx) + ".pdf", format="pdf", bbox_inches="tight")
        plt.show()