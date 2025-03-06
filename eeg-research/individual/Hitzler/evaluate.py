from EEGEvaluator import EEGEvaluator
import torch
from torch.utils.data import DataLoader
from EEGDataset import EEGDataset
from models.alexnet import AlexNet
from models.chrononet import ChronoNet
from models.cnn1d_lstm import CNN1D_LSTM
from models.cnn2d_lstm import CNN2D_LSTM
from models.cnn2d_blstm import CNN2D_BLSTM
from models.cnn1d_blstm import CNN1D_BLSTM
from models.densenet import DenseNet
from models.eegnet_pytorch import EEGNet
from models.mobilenet_lstm import MobileNet_LSTM
from models.resnet_dilation_lstm import Resnet_Dialation_LSTM
from models.resnet_lstm import ResNet_LSTM
from utils import read_json
from torch.utils.tensorboard import SummaryWriter

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
    "CNN1D_BLSTM": CNN1D_BLSTM

}

if __name__ == "__main__":
    # read configuration files
    config = read_json('configs/config.json')
    test_config = read_json('configs/test.json')
    valid_config = read_json('configs/valid.json')
    train_config = read_json('configs/train.json')
    config["batch_size"] = 1

    # load the model
    model = MODEL_DICT[test_config["model"]](config, torch.device(config["device"]))
    model.load_state_dict(torch.load(test_config["model_path"]))

    # create the dataset and dataloader
    train_dataset = EEGDataset(train_config["data_dir"], train_config["labels_dir"], config["enc_model"],
                               config["eeg_type"])
    valid_dataset = EEGDataset(valid_config["data_dir"], valid_config["labels_dir"], config["enc_model"],
                               config["eeg_type"])
    test_dataset = EEGDataset(test_config["data_dir"], test_config["labels_dir"], config["enc_model"],
                              config["eeg_type"])

    trainloader = DataLoader(train_dataset, batch_size=1,
                             num_workers=0, persistent_workers=False, drop_last=True, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=1,
                            num_workers=0, persistent_workers=False, drop_last=True, pin_memory=True)
    validloader = DataLoader(valid_dataset, batch_size=1,
                             num_workers=0, persistent_workers=False, drop_last=True, pin_memory=True)

    # create logger
    results_path = './evaluate'
    writer = SummaryWriter(log_dir=f"{results_path}/{model.__class__.__name__}")

    # create the evaluator
    eegevaluator = EEGEvaluator(model, config, torch.device(config["device"]), testloader, writer)

    # evaluate the model
    test_auc, test_acc, test_ap, test_tpr, test_tnr, avg_time, margin_onset, margin_offset, precision, recall = eegevaluator.evaluate_sliding_window()
    print(
        f"Test AUC: {test_auc}, Test ACC: {test_acc}, Test AP: {test_ap}, Test TPR: {test_tpr}, Test TNR: {test_tnr}, Average inference time: {avg_time} seconds, Margin Onset Accuracy: {margin_onset}, Margin Offset Accuracy: {margin_offset} Precision: {precision} Recall: {recall}")