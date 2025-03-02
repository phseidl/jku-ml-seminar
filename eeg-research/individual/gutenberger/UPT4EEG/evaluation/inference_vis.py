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
print(torch.cuda.get_device_name(device))

import os
print(os.environ.get("CUDA_VISIBLE_DEVICES"))

from torch import from_numpy as np2TT
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
from pathlib import Path


from UPT4EEG.utils.build import create_dataset
from UPT4EEG.evaluation.vis_eeg import viewARA, ar_through_model
from UPT4EEG.dataset.sparse_eeg_dataset_val import SparseEEGDataset_val
from UPT4EEG.dataset.collator_val import SparseEEG_Collator
from UPT4EEG_mp_2.model.encoder import Encoder
from UPT4EEG_mp_2.model.decoder import DecoderPerceiver
from UPT4EEG_mp_2.model.UPT4EEG import UPT4EEG


DATASET = 'TUH'    # either 'TUH' or 'BCI' or 'DenoiseNet'
MODEL_CLASS = 'UPT4EEG'
plot_sample = False

config_path = '/system/user/studentwork/gutenber/configs/config.yml'

model_name = yaml.safe_load(Path(config_path).read_text())['model_name']
cfg_dataset = yaml.safe_load(Path(config_path).read_text())['Dataset']
cfg_general = yaml.safe_load(Path(config_path).read_text())


SFREQ      = cfg_dataset["sfreq"]
normalize  = cfg_dataset["normalize"]
#use_montage = cfg_dataset['use_montage']
#use_montage = 'user_specific'
use_montage = 'tuh'
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


window_size = 1
stride = 0.5
saved_model_type = 'big_rand' #'small_tuh'





#model_path = '/system/user/studentwork/gutenber/upt-minimal/logs/TUH/UPT4EEG/UPT4EEG_Jan02_14-31-42_train.pth' #denoising trained on tuh
#model_path = '/system/user/studentwork/gutenber/upt-minimal/logs/TUH/UPT4EEG/UPT4EEG_Jan07_11-58-55_val.pth' #denoising trained on random, val on tuh
#model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan18_10-47-07_val.pth' 
#model_path = '/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan19_17-49-57_val.pth'  #trained with bin loss!!

#model_path = '/system/user/studentwork/gutenber/upt-minimal/upt-minimal/upt-minimal/logs/TUH/UPT4EEG/UPT4EEG_Dec30_16-14-33_val.pth' #recostruction
#'/system/user/studentwork/gutenber/upt-minimal/logs/TUH/UPT4EEG/UPT4EEG_Dec22_10-45-06_val.pth'


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


subject_vis = ["016"]
tmin = 500
tmax = 510

x_vis, y_vis, ch_names = create_dataset(
    os.path.join(cfg_dataset["x_basepath"], cfg_dataset["x_fpath"]),
    os.path.join(cfg_dataset["y_basepath"], cfg_dataset["y_fpath"]),
    [subject_vis], tmin=tmin, tmax=tmax,
    ch_names=cfg_dataset["ch_names"], win_size=window_size, stride=stride
)

#cfg_dataset["window_size"]
#x_train.shape: [Nr of segments, channel nr, sequence length]
x_vis = np2TT(x_vis)
y_vis = np2TT(y_vis)

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

query_vis = {'query_freq': 250.0, 'query_montage_pairs': query_montage_pairs}



    
vis_dataset = SparseEEGDataset_val(x_vis, y_vis, query_vis, ch_names, cfg_dataset, use_montage=use_montage)

vis_dataloader = DataLoader(
    dataset=vis_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=True,
    collate_fn=SparseEEG_Collator(num_supernodes=256, deterministic=False),
)


#noisy_data = x_vis.permute(1, 0, 2).reshape(len(ch_names), -1)
#reference_data = y_vis.permute(1, 0, 2).reshape(len(ch_names), -1)



os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
electrode = ['FP1-F7',
'm2',
'm3',
'm4',
'm5',
'm6',
'm7',
'm8',
'm9',
'T3-C3',
'C3-CZ',
'm12',
'C4-T4',
'm14',
'm15',
'm16',
'm17',
'm18',
'm19',
'P4-O2',
    ]

picks_chs = ['m11',
'm12',
'm13',
'm1',
'm20']

picks_chs = ['FP1-F7',
'T3-C3',
'C3-CZ',
'C4-T4',
'P4-O2',
    ]

#picks_chs = electrode


noisy_data, reference_data, reconstructed_data = ar_through_model(vis_dataloader, model, query_vis, window_size, stride, SFREQ, device)

    
print(f'Noisy data shape: {noisy_data.shape}')   
print(f'Reconstr. data shape: {reconstructed_data.shape}')

plt_interval = np.array([0, 10])
plt_interval_input = plt_interval*int(SFREQ)
plt_interval_query = plt_interval*int(query_vis['query_freq'])

start_input = plt_interval_input[0]
x_min_inp, x_max_inp = start_input, start_input + plt_interval_input[1]
start_query = plt_interval_query[0]
x_min_query, x_max_query = start_query, start_query + plt_interval_query[1]

x_data = np.array(noisy_data[:, x_min_inp: x_max_inp])
y_data = np.array(reference_data[:, x_min_inp: x_max_inp])
p_data = np.array(reconstructed_data[:, x_min_query: x_max_query])


fig, ax = plt.subplots(1, 1, figsize=(16, 9))
viewARA(
    np.linspace(0, math.ceil(x_data.shape[-1] / SFREQ), x_data.shape[-1]),
    np.linspace(0, math.ceil(p_data.shape[-1] / query_vis['query_freq']), p_data.shape[-1]),
    [x_data, y_data, y_data, p_data], 1, electrode, "CLEAN",
    titles=["Noisy", "", "Target", "CLEAN"], colors=["gray", "gray", "red", "blue"], alphas=[0.5, 0, 0.8, 0.8], ax=ax,
    picks_chs = picks_chs
)
plt.savefig("UPT4EEG/evaluation/plots/inference_S" + subject_vis[0] + "_" + model_weights_name+".pdf", format="pdf", bbox_inches="tight")
print("Saved to UPT4EEG/evaluation/plots/inference_S" + subject_vis[0] + "_" + model_weights_name+".pdf")
plt.show()


