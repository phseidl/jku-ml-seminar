import torch
from lightning import seed_everything
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from EEGTrainer import EEGTrainer
from EEGDataset import EEGDataset
from models.alexnet import AlexNet
from models.chrononet import ChronoNet
from models.cnn1d_lstm import CNN1D_LSTM
from models.cnn2d_lstm import CNN2D_LSTM
from models.densenet import DenseNet
from models.eegnet_pytorch import EEGNet
from models.transformer.guided_feature_transformer import GFT
from models.mobilenet_lstm import MobileNet_LSTM
from models.resnet_dilation_lstm import Resnet_Dialation_LSTM
from models.resnet_lstm import ResNet_LSTM
from models.vgg import VGG16
from utils import read_json

torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    seed_everything(42, workers=True)
    # read the configuration files
    config = read_json('configs/config.json')
    train = read_json('configs/train.json')
    valid = read_json('configs/valid.json')
    test = read_json('configs/test.json')

    # create the dataset
    train_dataset = EEGDataset(train["data_dir"], train["labels_dir"], config["enc_model"])
    valid_dataset = EEGDataset(valid["data_dir"], valid["labels_dir"], config["enc_model"])
    test_dataset = EEGDataset(test["data_dir"], test["labels_dir"], config["enc_model"])

    #df, counts = check_balance_of_dataset(train_dataset)
    # calculate loss weights
    #weight_0 = (counts[0] + counts[1]) / (2 * counts[0])
    #weight_1 = (counts[0] + counts[1]) / (2 * counts[1])
    #print(f"Weight for class 0: {weight_0}, Weight for class 1: {weight_1}")
#
    #class_weights = [weight_0, weight_1]
    #sample_weights = [class_weights[int(i)] for i in tqdm(df.labels.values, desc="Calculating sample weights")]
#
    #t_sampler = WeightedRandomSampler(
    #    weights=sample_weights, num_samples=len(train_dataset)
    #)
#
    results_path = './test'

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"],
                                              num_workers=0, persistent_workers=False, drop_last=True, pin_memory=True,
                                              sampler=None)
    # configuration for consine annealing scheduler
    config["steps_per_epoch"] = len(trainloader) * 27

    valloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False,
                                            num_workers=0, persistent_workers=False, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,
                                             num_workers=0, persistent_workers=False, drop_last=True, pin_memory=True)


    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    cnn1d = CNN1D_LSTM(config, device)
    cnn2d = CNN2D_LSTM(config, device)
    resnet = ResNet_LSTM(config, device)
    alexnext = AlexNet(config, device)
    transformer = GFT(config, device)
    vgg = VGG16(config, device)
    eegnet = EEGNet(config, device)
    chrononet = ChronoNet(config, device)
    densenet = DenseNet(config, device)
    mobilenet_lstm = MobileNet_LSTM(config, device)
    resnet_dialation_lstm = Resnet_Dialation_LSTM(config, device)


    for model in [eegnet]:
        writer = SummaryWriter(log_dir=f"{results_path}/{model.__class__.__name__}")
        trainer = EEGTrainer(model, config, device, trainloader, valloader, writer)
        trainer.train_sliding_window()
        writer.flush()
        writer.close()


