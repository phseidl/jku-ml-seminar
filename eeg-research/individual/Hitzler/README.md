Comparative Analysis of Deep Neural Networks for EEG-Based Real-Time Seizure Detection
========================================================================================
**Johannes Kepler University** - ML Seminar & Practical Work 2024/25  
**Supervisor**: Dr. Philipp Seidl, MSc \
**Author**: Florian Hitzler

The goal of this project is to compare the performance of different deep neural networks for real-time seizure detection using EEG Data based on the following paper:
[Real-Time Seizure Detection using EEG: A Comprehensive Comparison of Recent Approaches under a Realistic Setting](https://arxiv.org/abs/2201.08780)
There is also a code repository available from the authors (https://github.com/AITRICS/EEG_real_time_seizure_detection).
Unfortunately, the code uses a very old version of the [TUH EEG Dataset](https://isip.piconepress.com/projects/tuh_eeg/), which is not available anymore.

In my project I used the models from the original repository and adapted it to the current version (2.0.3) of the TUH EEG Dataset.
For feature extraction I used the two best performing methods in the paper (STFT and Frequency Bands) in addition to the raw EEG.

## Requirements
The environment can be set up using the provided `environment.yml` file with [anaconda](https://www.anaconda.com/). To create the environment, run the following command:
```
conda env create -f environment.yml -n <env_name>
```

## Data

The data used in this project is the [TUH EEG Seizure Corpus](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/).
To get the data, you need to request access to the corpus.
The current version of the dataset is 2.0.3.

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

The configuration files are stored in the `configs/` directory. The `config.json` file contains the hyperparameters for the models. 
The `train.json` and `valid.json` files contain the paths to the training and validation data respectively. 
The `test.json` file contains the paths to the test data.

| Parameter       | Description                                      | File                              |
|-----------------|--------------------------------------------------|-----------------------------------|
| device          | The device to use for training                   | config.json                       |
| batch_size      | The batch size to use for training               | config.json                       |
| epochs          | The number of epochs to train for                | config.json                       |
| lr              | The learning rate to use for training            | config.json                       |
| dropout         | The dropout rate to use for training             | config.json                       |
| num_channels    | The number of channels in the data               | config.json                       |
| num_classes     | The number of classes in the data                | config.json                       |
| sample_rate     | The sample rate of the data                      | config.json                       |
| val_interval    | The validation interval in percent               | config.json                       |
| enc_model       | The feature extraction model to use (raw, stft)  | config.json                       |
| log_interval    | The logging interval in number of steps          | config.json                       |
| models_to_train | The models which should be trained (class_names) | train.json                        |
| data_dir        | The directory containing the data                | train.json, valid.json, test.json |
| labels_dir      | The directory containing the labels              | train.json, valid.json, test.json |
| model_path      | The path to the model to evaluate                | test.json                         |
| model           | The name of the model to evaluate                | test.json                         |

## Usage

This section describes how to use the code in this repository in order to train and evaluate the models. 
The sections should be read in order, as the preprocessing step is required before training and evaluation.

### Preprocessing

The preprocessing script is used to preprocess the data.

```
python preprocess.py --data_path <path_to_data> --save_location <save_location> --channels <channel1, channel2> --alternative_channel_names <alternative_channel1, alternative_channel2>
```

The `--data_path` argument specifies the path to the data directory. 
The `--save_location` argument specifies the location where the preprocessed data should be saved. 
The `--channels` argument specifies the channels to use for preprocessing. All the channel names have to be present in the data.
The `--alternative_channel_names` argument specifies the alternative channel names to use if the first set of channels don't exist.

Here is an example of how to run the preprocessing script:
```
python preprocess.py --data_path /path/to/data --save_location /path/to/save/location --channels C3, C4, C5, C6 --alternative_channel_names C3A, C4A, C5A, C6A
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

```
python train.py
```

### Evaluation

The evaluation script can be used to evaluate the models. It requires the configuration of test.json


```
python evaluate.py
```

