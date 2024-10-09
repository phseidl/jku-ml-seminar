import json

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils import data
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from utils import check_balance_of_dataset, read_json

from EEGDataset import EEGDataset
from LightningCNN import LightningCNN
from models.cnn2d_lstm import CNN2D_LSTM_V8
from models.cnn1d_lstm import CNN1D_LSTM_V8
from models.alexnet import ALEXNET_V4
from models.resnet_lstm import Resnet
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler, PyTorchProfiler

torch.set_float32_matmul_precision('medium')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    seed = torch.Generator().manual_seed(42)
    config = read_json('configs/config.json')
    train = read_json('configs/train.json')
    valid = read_json('configs/valid.json')
    test = read_json('configs/test.json')

    train_dataset = EEGDataset(train["data_dir"], train["labels_dir"])
    valid_dataset = EEGDataset(valid["data_dir"], valid["labels_dir"])
    # split the train set into two
    test_dataset = EEGDataset(test["data_dir"], test["labels_dir"])

    df, counts = check_balance_of_dataset(train_dataset)
    check_balance_of_dataset(valid_dataset, subset=False, type='valid')
    check_balance_of_dataset(test_dataset, subset=False, type='test')
    # calculate loss weights
    weight_0 = (counts[0] + counts[1]) / (2 * counts[0])
    weight_1 = (counts[0] + counts[1]) / (2 * counts[1])
    print(f"Weight for class 0: {weight_0}, Weight for class 1: {weight_1}")

    class_weights = [weight_0, weight_1]
    sample_weights = [class_weights[int(i)] for i in tqdm(df.labels.values, desc="Calculating sample weights")]

    t_sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_dataset)
    )

    results_path = './tensorboard'

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"],
                                              shuffle=config["shuffle"],
                                              num_workers=4, persistent_workers=True, drop_last=True, pin_memory=True,
                                              sampler=t_sampler)
    valloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False,
                                            num_workers=4, persistent_workers=True, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,
                                             num_workers=4, persistent_workers=True, drop_last=True, pin_memory=True)

    # TODO: Merge with config
    args = {"num_layers": 2, "batch_size": config["batch_size"], "dropout": 0.0, "num_channel": 19, "enc_model": "raw",
            "output_dim": 1, "sincnet_bandnum": 10, "eeg_type": "bipolar", "sincnet_layer_num": 3, "window_size": 4,
            "window_shift": 1,
            "sample_rate": 200, "sincnet_kernel_size": 3, "sincnet_stride": 1,
            "feature_sample_rate": 200, "sincnet_input_normalize": None, "lr_init": 0.001,
            "steps_per_epoch": len(trainloader) * 27,
            "epochs": config["epochs"]}
    args["cnn_channel_sizes"] = [args["sincnet_bandnum"], 10, 10]
    args["window_size_label"] = args["feature_sample_rate"] * args["window_size"]
    args["window_shift_label"] = args["feature_sample_rate"] * args["window_shift"]
    args["window_size_sig"] = args["sample_rate"] * args["window_size"]
    args["window_shift_sig"] = args["sample_rate"] * args["window_shift"]

    cnn1d = LightningCNN(CNN1D_LSTM_V8(args, device), args)
    cnn2d = LightningCNN(CNN2D_LSTM_V8(args, device), args)
    resnet = LightningCNN(Resnet(args, device), args)
    alexnext = LightningCNN(ALEXNET_V4(args, device), args)


    for model in [cnn2d]:
        logger = TensorBoardLogger(results_path, name=model.model.__class__.__name__)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer = L.Trainer(min_epochs=1, max_epochs=config["epochs"], accelerator=device.type, logger=logger,
                            val_check_interval=0.5, callbacks=[lr_monitor])
        trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)
        trainer.test(model, dataloaders=testloader)