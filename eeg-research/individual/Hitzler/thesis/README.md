Comparative Analysis of Deep Neural Networks for EEG-Based Real-Time Seizure Detection
========================================================================================
**Johannes Kepler University** - Bachelor's Thesis 2025 \
**Supervisor**: Dr. Philipp Seidl, MSc \
**Author**: Florian Hitzler

The goal of this project is to compare the performance of different deep neural networks for real-time seizure detection using EEG Data.

## Requirements
The environment can be set up using the provided `environment.yml` file with [anaconda](https://www.anaconda.com/). To create the environment, run the following command:
```
conda env create -f environment.yml -n <env_name>
```

**INFO**: The environment only runs on Linux (as the Mamba models are not supported on Windows).
The env uses Python 3.9 with cuda 11.8.

## Data

The data used in this project is the [TUH EEG Seizure Corpus](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/) and [CHB-MIT Scalp EEG Database
](https://physionet.org/content/chbmit/1.0.0/).
To get the TUH dataset, one needs to request access to the corpus.
The current version of the dataset is 2.0.3, which is also used in this project.
## Code Structure

The code is structured as follows:
- `configs/`: Contains the configuration files.
- `models/`: Contains the model implementations.
- `scheduler/`: Contains learning rate scheduler implementations.
- `EEGDataset.py`: Contains the dataset implementation.
- `EEGEvalutor.py`: Contains the evaluation class.
- `EEGTrainer.py`: Contains the training class.
- `evaluate.py`: Contains the evaluation code.
- `preprocessing.py`: Contains the preprocessing code.
- `train.py`: Contains the training code.
- `utils.py`: Contains utility functions.


## Configuration

The configuration files are stored in the `configs/` directory. The `config.json` file contains global hyperparameters for the models. \
The `train.json` and `valid.json` files contain the paths to the training and validation data respectively. \
The `test.json` file contains the paths to the test data.

| Parameter       | Description                                                                    | File                              |
|-----------------|--------------------------------------------------------------------------------|-----------------------------------|
| device          | The device to use for training                                                 | config.json                       |
| batch_size      | The batch size to use for training                                             | config.json                       |
| epochs          | The number of epochs to train for                                              | config.json                       |
| lr_init         | The learning rate to use for training                                          | config.json                       |
| dropout         | The dropout rate                                                               | config.json                       |
| num_layers      | The number of lstm layers (for lstm based models)                              | config.json                       |
| sample_rate     | The sample rate of the data (as specified in preprocessing.py, default is 200) | config.json                       |
| val_interval    | The validation interval given as decimal percent                               | config.json                       |
| log_interval    | The logging interval in number of steps                                        | config.json                       |
| models_to_train | The models which should be trained (class_names)                               | train.json                        |
| data_dir        | The directory containing the data                                              | train.json, valid.json, test.json |
| labels_dir      | The directory containing the labels                                            | train.json, valid.json, test.json |
| model_path      | The path to the model to evaluate                                              | test.json                         |
| model           | The name of the model to evaluate                                              | test.json                         |

## Usage

This section describes how to use the code in this repository in order to train and evaluate the models. 
The sections should be read in order, as the preprocessing step is required before training and evaluation.

### Preprocessing

The preprocessing scripts are used to preprocess the data. There are two preprocessing scripts available: one for the TUH dataset and one for the CHB-MIT dataset.

Here are examples of how to run the preprocessing scripts for both datasets:

#### TUH Dataset
```
python preprocessingTUH.py --data_dir <path_to_data> --save_location <save_location> --channels <channel1, channel2> --alternative_channel_names <alternative_channel1, alternative_channel2>
```

The `--data_dir` argument specifies the path to the data directory. \
The `--save_location` argument specifies the location where the preprocessed data should be saved. \
The `--channels` argument specifies the channels to use for preprocessing. All the channel names have to be present in the data.\
The `--alternative_channel_names` argument specifies the alternative channel names to use if the first set of channels don't exist.

Here is an example of how to run the preprocessing script:
```
python preprocessingTUH.py --data_dir "data/TUH/2.0.3/raw/dev" --save_location "data/TUH/2.0.3/processed/dev" --channels "EEG FP1-REF, EEG FP2-REF, EEG F3-REF, EEG F4-REF, EEG C3-REF, EEG C4-REF, EEG P3-REF, EEG P4-REF, EEG O1-REF, EEG O2-REF, EEG F7-REF, EEG F8-REF, EEG T3-REF, EEG T4-REF, EEG T5-REF, EEG T6-REF, EEG CZ-REF, EEG PZ-REF, EEG FZ-REF" --alternative_channel_names "EEG FP1-LE, EEG FP2-LE, EEG F3-LE, EEG F4-LE, EEG C3-LE, EEG C4-LE, EEG P3-LE, EEG P4-LE, EEG O1-LE, EEG O2-LE, EEG F7-LE, EEG F8-LE, EEG T3-LE, EEG T4-LE, EEG T5-LE, EEG T6-LE, EEG CZ-LE, EEG PZ-LE, EEG FZ-LE"
```

The script should be run for each of the three datasets (train, dev/valid, test) separately.

#### CHB-MIT Dataset

The preprocessing script for the CHB-MIT dataset is similar, but it does not require alternative channel names as the dataset has a fixed set of channels.

```
python preprocessingMIT.py --data_dir <path_to_data> --save_location <save_location> --channels <channel1, channel2> 
```

The `--data_dir` argument specifies the path to the data directory.\
The `--save_location` argument specifies the location where the preprocessed data should be saved. \
The `--channels` argument specifies the channels to use for preprocessing. All the channel names have to be present in the data.

Here is an example of how to run the preprocessing script:
```
python preprocessingMIT.py --data_dir "data/mit/raw" --save_location "data/mit/processed" --channels "FP1-F7, F7-T7, T7-P7, P7-O1, FP1-F3, F3-C3, T8-P8-0, C3-P3, P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2, FP2-F8, F8-T8, P8-O2, FZ-CZ, CZ-PZ, P7-T7, T7-FT9"
```



### Training

For the training script the config.json file can be used to specify the hyperparameters as well as the dataset locations via train.json and valid.json respectively.
One can also specify the models to train in the train.json file ("models_to_train").
Valid options are:
- `EEGNet`
- `AlexNet`
- `ChronoNet`
- `CNN2D_LSTM`
- `CNN1D_LSTM`
- `DenseNet`
- `MobileNet_LSTM`
- `Resnet_Dialation_LSTM`
- `ResNet_LSTM`
- `CNN2D_BLSTM`
- `CNN1D_BLSTM`
- `ResNet`
- `MobileNetV3`
- `TDNN_LSTM`
- `FeatureTransformer`
- `GuidedFeatureTransformer`
- `BASE_MAMBA`
- `BASE_MAMBA2`
- `BASE_XLSTM`
- `CNN_MAMBA`
- `CNN_MAMBA2`
- `CNN_XLSTM`

A default configuration file for the CNN2D_LSTM is provided in the `configs/` directory.

```
python train.py
```

### Evaluation

The evaluation script can be used to evaluate the models. It requires the configuration of test.json.

Here also a default configuration file for the CNN2D_LSTM is provided in the `configs/` directory for the TUH dataset.
The path can be changed to the MIT processed path in the test.json file to evaluate the models on the CHB-MIT dataset.

```
python evaluate.py
```

