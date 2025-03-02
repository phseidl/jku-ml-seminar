# check successful setup
import torch
from torch import from_numpy as np2TT
from torch.utils.data import DataLoader
from datetime import datetime
import yaml
from pathlib import Path
import os
import argparse

print("Current working directory:", os.getcwd())

from UPT4EEG.utils.build import create_dataset
from UPT4EEG.dataset.sparse_eeg_dataset import SparseEEGDataset
from UPT4EEG.dataset.sparse_eeg_dataset_usz import SparseEEGDataset_USZ
from UPT4EEG.model.position_encoding import ContinuousSincosEmbed, ChannelPositionalEncodingSinCos
from UPT4EEG.model.encoder import Encoder
from UPT4EEG.model.decoder import DecoderPerceiver
from UPT4EEG.model.UPT4EEG import UPT4EEG
from UPT4EEG.utils.train import Trainer
from UPT4EEG.utils.model_tracer import Model_Tracer
from UPT4EEG.dataset.collator import SparseEEG_Collator


# CUDA
if torch.cuda.is_available():
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Using CPU.")
print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))
print("Device name: " + str(torch.cuda.get_device_name(device)))
print(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES")}')


######################## CONFIGS ########################
# Create the parser
parser = argparse.ArgumentParser(description="Main UPT4EEG training script.")

# Define the argument
parser.add_argument('config_file', type=str, help="Config file (.yml): relative path in working directory.")

# Parse the arguments
args = parser.parse_args()

config_file = args.config_file

config_path = os.path.join(os.getcwd(), config_file)
#config_path = '/system/user/studentwork/gutenber/configs/config.yml'
#model_config_path = '/content/drive/My Drive/A_EEG/CLEEGN/configs/tusz/model_config.yml'

model_name = yaml.safe_load(Path(config_path).read_text())['model_name']
cfg_dataset = yaml.safe_load(Path(config_path).read_text())['Dataset']
cfg_general = yaml.safe_load(Path(config_path).read_text())
#cfg_model = yaml.safe_load(Path(model_config_path).read_text())[MODEL_CLASS]

MODEL_CLASS = cfg_general['model_class']
DATASET = cfg_dataset['dataset'] 
hyperparams = cfg_general['Hyperparameters']
SFREQ      = cfg_dataset["sfreq"]
normalize  = cfg_dataset["normalize"]
#use_montage = cfg_dataset['use_montage']
use_montage = cfg_dataset['use_montage']
use_montage_val = cfg_dataset['use_montage_val']
#NUM_EPOCHS = 1 #cfg_general['epochs']
#BATCH_SIZE = cfg_model['batch_size']
#LR         = cfg_model["learning_rate"]
try:
    loss_fun = hyperparams['loss_fun']
except:
    loss_fun = 'mse'

if loss_fun == 'bin':
    from UPT4EEG.utils.train_bin import Trainer
elif loss_fun == 'ensemble':
    from UPT4EEG.utils.train_ensemble import Trainer

SAVE_PATH = 'logs/' + DATASET + '/' + MODEL_CLASS

if not os.path.exists(SAVE_PATH):
    try:
        os.makedirs(SAVE_PATH)
    except Exception as e:
        print(f"Failed to create directory '{SAVE_PATH}': {e}")


timestamp = datetime.now().strftime("%b%d_%H-%M-%S")

######################## LOAD EEG DATASET ########################
window_size = cfg_dataset['window_size']
stride = cfg_dataset['stride']
input_chs = cfg_dataset['input_chs']
output_chs = cfg_dataset['output_chs']
num_inputs = cfg_dataset['num_inputs']
num_outputs = cfg_dataset['num_outputs']
ch_dropout = cfg_dataset['ch_dropout']
io_same = cfg_dataset['io_same']
train = True  # Random sampling for training

if DATASET == 'TUH':
    x_train, y_train, ch_names = create_dataset(
        os.path.join(cfg_dataset["x_basepath"], cfg_dataset["x_fpath"]),
        os.path.join(cfg_dataset["y_basepath"], cfg_dataset["y_fpath"]),
        cfg_dataset["subjects_train"], tmin=cfg_dataset["tmin"], tmax=cfg_dataset["tmax"],
        ch_names=cfg_dataset["ch_names"], win_size=window_size, stride=stride
    )

    #cfg_dataset["window_size"]
    #x_train.shape: [Nr of segments, channel nr, sequence length]
    x_train = np2TT(x_train)
    y_train = np2TT(y_train)


    x_valid, y_valid, ch_names = create_dataset(
        os.path.join(cfg_dataset["x_basepath"], cfg_dataset["x_fpath"]),
        os.path.join(cfg_dataset["y_basepath"], cfg_dataset["y_fpath"]),
        cfg_dataset["subjects_val"], tmin=cfg_dataset["tmin"], tmax=cfg_dataset["tmax"],
        ch_names=cfg_dataset["ch_names"], win_size=window_size, stride=stride)

    x_valid = np2TT(x_valid)
    y_valid = np2TT(y_valid)


    train_dataset = SparseEEGDataset(x_train, 
                                    y_train, 
                                    num_inputs, 
                                    num_outputs, 
                                    ch_names, 
                                    cfg_dataset, 
                                    train, 
                                    use_montage=use_montage, 
                                    n_input_chs=input_chs, 
                                    n_output_chs=output_chs, 
                                    ch_dropout=ch_dropout,
                                    io_same = io_same)

    val_dataset = SparseEEGDataset(x_valid, 
                                y_valid, 
                                num_inputs, 
                                num_outputs, 
                                ch_names, 
                                cfg_dataset, 
                                train = False, 
                                use_montage=use_montage_val, 
                                io_same = True)
elif DATASET == 'USZ':
    downsample_ieeg = 200
    train = True  # Random sampling for training
    #eeg_channels_to_keep = ['C4', 'O1', 'C3', 'A1', 'A2', 'F3', 'F4', 'O2']
    eeg_channels_to_keep_s5 = ['Fp1', 'Fp2', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'F7', 'F8', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'A1', 'A2']
    #ieeg_channels_to_keep = ['mAHL4', 'mAL3', 'mAL1', 'mAHL5', 'mAHL1', 'mAHL3', 'mAL4', 'mAL5', 'mAHL8', 'mAL6', 'mAL7', 'mAL2', 'mAHL7', 'mAL8', 'mAHL6']
    eeg_channels_to_keep = None
    ieeg_channels_to_keep = None

    subjects_train = ['02', '03', '04', '07', '09', '11'] # ['02', '03', '04', '07', '09', '11', '15'] #['04']   #, '02', '03', '04'
    subjects_val = ['05']
    window_size = 1.0
    stride = 1.0

    train_dataset = SparseEEGDataset_USZ(subjects_train,
                                        window_size, 
                                        stride, 
                                        num_inputs, 
                                        num_outputs, 
                                        cfg_dataset, 
                                        eeg_channels_to_keep, 
                                        ieeg_channels_to_keep, 
                                        train, 
                                        downsample_ieeg = downsample_ieeg, 
                                        use_montage=use_montage, 
                                        n_input_chs=input_chs, 
                                        n_output_chs=output_chs, 
                                        ch_dropout=ch_dropout)
    #val_dataset = train_dataset
    val_dataset = SparseEEGDataset_USZ(subjects_val, 
                                    window_size, 
                                    stride, 
                                    num_inputs, 
                                    num_outputs, 
                                    cfg_dataset, 
                                    eeg_channels_to_keep_s5, 
                                    ieeg_channels_to_keep, train, 
                                    downsample_ieeg = downsample_ieeg, 
                                    use_montage=use_montage, 
                                    n_input_chs=input_chs, 
                                    n_output_chs=output_chs, 
                                    ch_dropout=ch_dropout)


sample = train_dataset[0]
print(f"Input features shape: {sample['input_feat'].shape}")
print(f"Input positions shape: {sample['input_pos'].shape}")
print(f"Target features shape: {sample['target_feat'].shape}")
print(f"Output positions shape: {sample['output_pos'].shape}")



def main_fn(train_dataset, val_dataset, config, device):
    load_checkpoint = False
    model_path = None #'/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/UPT4EEG_Jan19_18-47-03_val.pth'  #small model on tuh  #'/content/drive/My Drive/A_EEG/CLEEGN/logs/USZ/UPT4EEG/UPT4EEG_Dec11_11-42-17.pth'

    use_wandb = True
    hyperparams = config['Hyperparameters']
    # hyperparameters
    d_model = hyperparams['d_model']
    dim = hyperparams['dim']   # ~6M parameter model
    num_heads = hyperparams['num_heads']
    depth = hyperparams['depth']
    epochs = hyperparams['epochs']
    batch_size = hyperparams['batch_size']
    lr = hyperparams['lr']
    optim_weight_decay = hyperparams['optim_weight_decay']
    linear_decay_end = hyperparams['linear_decay_end']
    warmup_perc = hyperparams['warmup_perc']
    input_dim = 1
    output_dim = 1
    use_mlp_posEnc = True
    accumulation_steps = 1
    num_supernodes = 512

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

    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    if load_checkpoint:
        state_path = os.path.join(model_path)
        state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state["state_dict"])

    # setup dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=SparseEEG_Collator(num_supernodes=num_supernodes, deterministic=False),
        num_workers=4,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        collate_fn=SparseEEG_Collator(num_supernodes=num_supernodes, deterministic=True),
    )

    ckpts = [
        Model_Tracer(monitor="loss", mode="min", do_save=True, root=SAVE_PATH, prefix= MODEL_CLASS + '_' + timestamp + '_train'),
        Model_Tracer(monitor="val_loss", mode="min", do_save=True, root=SAVE_PATH, prefix= MODEL_CLASS + '_' + timestamp+ '_val'),
    ]

    # Loss function
    loss_fn = torch.nn.MSELoss()

    # initialize optimizer and learning rate schedule (linear warmup for first 10% -> linear decay)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=optim_weight_decay)
    total_updates = len(train_dataloader) * epochs * accumulation_steps
    warmup_updates = int(total_updates * warmup_perc)
    lrs = torch.concat(
        [
            # linear warmup
            torch.linspace(0, optim.defaults["lr"], warmup_updates),
            # linear decay
            torch.linspace(optim.defaults["lr"], linear_decay_end, total_updates - warmup_updates),
        ],
    )

    wandb_config = {
            "config": config,
            "initial_learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "updates": total_updates,
            "d_model": d_model,
            "dim": dim,
            "num_heads": num_heads,
            "num_supernodes": num_supernodes,
            "transformer_depths": depth,
            "dataset": DATASET,
            "model_weights_timestamp": timestamp,
            "sample_num_inputs": num_inputs,
            "sample_num_outputs": num_outputs,
            "subjects_train": cfg_dataset["subjects_train"],
            "subjects_val": cfg_dataset["subjects_val"],
            "window_size": window_size,
            "stride": stride,
            "optim_weight_decay": optim_weight_decay,
            "linear_decay_end": linear_decay_end,
            "warmup_perc": warmup_perc,
            "load_checkpoint": load_checkpoint,
            "model_path": model_path,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "use_mlp_posEnc": use_mlp_posEnc,
            "use_montage": use_montage,
            "use_montage_val": use_montage_val,
            "timestamp": timestamp,
            "accumulation_steps": accumulation_steps,
            "loss_fun": loss_fun,
        }

    train_config = {
        "lrs": lrs,
        "total_updates": total_updates,
        "ckpts": ckpts,
        "wandb_config": wandb_config,
        "accumulation_steps": accumulation_steps,
    }


    trainer = Trainer(model, optim, loss_fn, device, use_wandb, 'UPT4EEG')
    trainer.train(train_dataloader, val_dataloader, epochs, train_config)


if __name__ == "__main__":
    main_fn(train_dataset, val_dataset, cfg_general, device)