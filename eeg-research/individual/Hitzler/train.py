import torch
from lightning import seed_everything
from torch.utils import data

from EEGDataset import EEGDataset
from torch.utils.tensorboard import SummaryWriter

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
from models.transformer.feature_transformer import FT
from models.transformer.guided_feature_transformer import GFT
from models.vgg import VGG16
from models.tdnn_lstm import TDNN_LSTM
from utils import read_json, check_balance_of_dataset, compute_mean_std


from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler

torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    seed_everything(42, workers=True)
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

    cnn1d = CNN1D_LSTM(config, device)
    bcnn1d = CNN1D_BLSTM(config, device)
    bcnn2d = CNN2D_BLSTM(config, device)
    cnn2d = CNN2D_LSTM(config, device)
    resnet2d = RESNET18_CONV2D(config, device)
    alexnet = AlexNet(config, device)
    mobilenet = MobileNetV3(config, device)
    guided_transformer = GFT(config, device)
    vgg = VGG16(config, device)
    eegnet = EEGNet(config, device)
    chrononet = ChronoNet(config, device)
    densenet = DenseNet(config, device)
    mobilenet_lstm = MobileNet_LSTM(config, device)
    resnet_dialation_lstm = Resnet_Dialation_LSTM(config, device)
    transformer = FT(config, device)
    tdnn_lstm = TDNN_LSTM(config, device)
    resnet_lstm = ResNet_LSTM(config, device)
    for model in [bcnn1d]:
        writer = SummaryWriter(log_dir=f"{results_path}/{model.__class__.__name__}")
        trainer = EEGTrainer(model, config, device, trainloader, valloader, writer)
        trainer.train_sliding_window()
        writer.flush()
        writer.close()


