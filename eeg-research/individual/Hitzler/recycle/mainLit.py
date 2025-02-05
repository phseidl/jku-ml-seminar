import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import lightning as L
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from torch.utils import data

from EEGDataset import EEGDataset
from LightningCNN import LightningCNN
from individual.Hitzler.InferenceBenchmark import InferenceBenchmark
from individual.Hitzler.models.eegnet import EEGNet
from individual.Hitzler.models.guided_feature_transformer import EEG_FEATURE_TRANSFORMER_V15_GCT
from individual.Hitzler.models.vgg import VGG16
from models.alexnet import ALEXNET_V4
from models.cnn2d_lstm import CNN2D_LSTM_V8
from models.resnet_lstm import Resnet
from utils import check_balance_of_dataset, read_json

torch.set_float32_matmul_precision('medium')
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    seed_everything(42, workers=True)
    config = read_json('configs/config.json')
    train = read_json('configs/train.json')
    valid = read_json('configs/valid.json')
    test = read_json('configs/test.json')

    train_dataset = EEGDataset(train["data_dir"], train["labels_dir"])
    valid_dataset = EEGDataset(valid["data_dir"], valid["labels_dir"])
    # split the train set into two
    test_dataset = EEGDataset(test["data_dir"], test["labels_dir"])

    #df, counts = check_balance_of_dataset(train_dataset)
    #check_balance_of_dataset(valid_dataset, subset=False, type='valid')
    #check_balance_of_dataset(test_dataset, subset=False, type='test')
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

    results_path = './tensorboard'

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"],
                                              shuffle=config["shuffle"],
                                              num_workers=0, persistent_workers=False, drop_last=True, pin_memory=True)
    config["steps_per_epoch"] = len(trainloader) * 27
    
    valloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False,
                                            num_workers=0, persistent_workers=False, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,
                                             num_workers=0, persistent_workers=False, drop_last=True, pin_memory=True)


    #cnn1d = LightningCNN(CNN1D_LSTM(config, device), config)
    cnn2d = LightningCNN(CNN2D_LSTM_V8(config, device), config)
    resnet = LightningCNN(Resnet(config, device), config)
    alexnext = LightningCNN(ALEXNET_V4(config), config)
    transformer = LightningCNN(EEG_FEATURE_TRANSFORMER_V15_GCT(config, device), config)
    vgg = LightningCNN(VGG16(config, device), config)
    eegnet = LightningCNN(EEGNet(config), config)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        every_n_train_steps=2500
    )
    #benchmark_callback = InferenceBenchmark()
    profiler = PyTorchProfiler(dirpath=".", filename="perf_logs_pytorch")
    for model in [cnn2d]:
        logger = TensorBoardLogger(results_path, name=model.model.__class__.__name__)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer = L.Trainer(min_epochs=1, max_epochs=config["epochs"], accelerator="cpu", logger=logger,
                            callbacks=[lr_monitor], log_every_n_steps=25, fast_dev_run=True)
        trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)
        #trainer.test(model, dataloaders=testloader)