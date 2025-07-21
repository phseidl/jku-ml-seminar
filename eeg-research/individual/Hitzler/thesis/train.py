import os
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from EEGDataset import EEGDataset
from EEGTrainer import EEGTrainer
from models.alexnet import AlexNet
from models.chrononet import ChronoNet
from models.cnn1d_blstm import CNN1D_BLSTM
from models.cnn1d_lstm import CNN1D_LSTM
from models.cnn2d_blstm import CNN2D_BLSTM
from models.cnn2d_lstm import CNN2D_LSTM
from models.densenet import DenseNet
from models.eegnet_pytorch import EEGNet
from models.mobilenet import MobileNetV3
from models.mobilenet_lstm import MobileNet_LSTM
from models.resnet import RESNET18_CONV2D
from models.resnet_dilation_lstm import Resnet_Dialation_LSTM
from models.resnet_lstm import ResNet_LSTM
from models.tdnn_lstm import TDNN_LSTM
from models.transformer.feature_transformer import FT
from models.transformer.guided_feature_transformer import GFT
from models.vgg import VGG16
from utils import read_json, check_balance_of_dataset

torch.set_float32_matmul_precision('medium')

# dictionary to map model name to model class
MODEL_DICT = {
    "EEGNet": EEGNet,
    "AlexNet": AlexNet,
    "ChronoNet": ChronoNet,
    "CNN2D_LSTM": CNN2D_LSTM,
    "CNN1D_LSTM": CNN1D_LSTM,
    "DenseNet": DenseNet,
    "MobileNet_LSTM": MobileNet_LSTM,
    "Resnet_Dialation_LSTM": Resnet_Dialation_LSTM,
    "ResNet_LSTM": ResNet_LSTM,
    "CNN2D_BLSTM": CNN2D_BLSTM,
    "CNN1D_BLSTM": CNN1D_BLSTM,
    "ResNet": RESNET18_CONV2D,
    "MobileNetV3": MobileNetV3,
    "TDNN_LSTM": TDNN_LSTM,
    "VGG": VGG16,
    "FeatureTransformer": FT,
    "GuidedFeatureTransformer": GFT
}

def convert_string_to_model(model_name, config, device):
    if model_name in MODEL_DICT:
        return MODEL_DICT[model_name](config, device)
    else:
        raise ValueError(f"Model {model_name} not found")

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    # read the configuration files
    config = read_json('configs/config.json')
    train = read_json('configs/train.json')
    valid = read_json('configs/valid.json')
    test = read_json('configs/test.json')



    train_dataset = EEGDataset(train["data_dir"], train["labels_dir"], config["enc_model"], config["eeg_type"])
    valid_dataset = EEGDataset(valid["data_dir"], valid["labels_dir"], config["enc_model"], config["eeg_type"])
    test_dataset = EEGDataset(test["data_dir"], test["labels_dir"], config["enc_model"], config["eeg_type"])

    df, counts = check_balance_of_dataset(train_dataset)

    # calculate loss weights
    weight_0 = 1 / counts[0]
    weight_1 = 1 / counts[1]
    print(f"Weight for class 0: {weight_0}, Weight for class 1: {weight_1}")

    class_weights = [weight_0, weight_1]
    sample_weights = [class_weights[int(i)] for i in tqdm(df.labels.values, desc="Calculating sample weights")]

    t_sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights)
    )

    results_path = './tensorboard'

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"],
                                              num_workers=4, persistent_workers=True, drop_last=True, pin_memory=True,
                                              sampler=t_sampler, shuffle=False)

    valloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False,
                                            num_workers=4, persistent_workers=True, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,
                                             num_workers=4, persistent_workers=True, drop_last=True, pin_memory=True)


    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    models = train["models_to_train"]

    # create the best_models folder if it does not exist
    os.makedirs("best_models", exist_ok=True)


    for model_class in models:
        model = convert_string_to_model(model_class, config, device)
        print(f"Training {model.__class__.__name__}...")
        writer = SummaryWriter(log_dir=f"{results_path}/{model.__class__.__name__}")
        trainer = EEGTrainer(model, config, device, trainloader, valloader, writer)
        trainer.train_sliding_window()
        writer.flush()
        writer.close()


