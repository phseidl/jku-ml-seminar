Seizure forecasting using wearable devices
==================================
**Johannes Kepler University** - ML Seminar & Practical Work 2023/24  

*Author: Jozsef Kovacs*  
*Created: 13/01/2023*  
*Modified: 06/05/2024*

---

The purpose of this project is to reproduce the deep learning approach to seizure forecasting from non-EEG signals,
collected from wrist-worn wearable devices that collect physiological patient parameters - as reported in the study
article:
    [Nasseri, M., Pal Attia, T., Joseph, B., Gregg, N. M., Nurse, E. S., Viana, P. F., ... & Brinkmann, B. H. (2021).
    Ambulatory seizure forecasting with a wrist-worn device using long-short term memory deep learning.
    Scientific reports, 11(1), 21935.](https://www.nature.com/articles/s41598-021-01449-2)

The full reproduction of the experiments was not possible, as the dataset is only partially available (duration of
monitoring, number of seizures). Furthermore, the above-mentioned paper did not have any acompanying codebase and
the authors did provide only general description about the implementation of preprocessing, training, testing.
Consequently, after analysis and a number of preliminary explorative experiments with the obtained dataset, 
this project was built from scratch following  the descriptions and indications from the article. 
The dataset was obtained from: https://www.epilepsyecosystem.org/

## Datasets

If you are interested in availability of wearable, non-EEG datasets for seizure prediction and forecasting, 
this document contains a brief overview: [WEARABLE_DATA.md](../../data/WEARABLE_DATA.md)   

### My Seizure Gauge dataset  

The following data is available from the Seer Medical portal after registration and authorisation by the 
[Epilepsy Ecosystem](https://www.epilepsyecosystem.org/). There are subjects with varying durations of
monitoring. The *Annotations* column contains the number of seizures (in total, which might not all be 
suitable to use as *lead seizures*). The last four columns specify the type of wearable devices that were
used for data acquisition. In this project, we are interested in the *E4* (Empatica E4), however the data
download process will also provide the Byteflies (*BF*) and *Epilog* data, where available.   


| Study name (subject)                                                                       | Total collected data (during monitoring)                                                              | Annotations | E4 | Byteflies EmgAcc | Byteflies AccPpg | Epilog  |
|--------------------------------------------------------------------------------------------| ----------------------------------------------------------------------------------------------------- |------------:|:--:|:----------------:|:----------------:|:-------:|
| [MSEL_01828](https://app.seermedical.com/studies/9ef709ff-51f6-493f-aabb-3fa8eb3eca12)     | [21 hr 24 min 43 s](https://app.seermedical.com/studies/9ef709ff-51f6-493f-aabb-3fa8eb3eca12)         |           1 | Y  |        N         |        N         |    N    |
| [MSEL_01709](https://app.seermedical.com/studies/f12d339c-9e10-4990-a22f-daf026c3fd71)     | [4 days, 18 hr 7 min 52 s](https://app.seermedical.com/studies/f12d339c-9e10-4990-a22f-daf026c3fd71)  |           6 | Y  |        N         |        N         |    N    |
| [MSEL_00095](https://app.seermedical.com/studies/ab50e3dd-4901-4efb-a08a-bc4123fd793d)     | [6 days, 22 hr 40 min 26 s](https://app.seermedical.com/studies/ab50e3dd-4901-4efb-a08a-bc4123fd793d) |           3 | Y  |        N         |        N         |    N    |
| [MSEL_01110-ICU](https://app.seermedical.com/studies/ab1d9a55-190a-431f-ad31-653e137f584c) | [3 days, 12 hr 18 min 39 s](https://app.seermedical.com/studies/ab1d9a55-190a-431f-ad31-653e137f584c) |          12 | Y  |        N         |        N         |    N    |
| [MSEL_01839](https://app.seermedical.com/studies/339e629a-588a-4581-8bca-339202ae1c46)     | [9 days, 21 hr 4 min 5 s](https://app.seermedical.com/studies/339e629a-588a-4581-8bca-339202ae1c46)   |           2 | N  |        Y         |        N         |    N    |
| [MSEL_00764](https://app.seermedical.com/studies/d56246f3-aa34-444d-9d95-28920c1e3507)     | [2 days, 21 hr 8 min 2 s](https://app.seermedical.com/studies/d56246f3-aa34-444d-9d95-28920c1e3507)   |           6 | Y  |        Y         |        N         |    N    |
| [MSEL_01462](https://app.seermedical.com/studies/6dc191c3-45c0-4990-bdf0-8dc6060d1f6e)     | [2 days, 18 hr 51 min 44 s](https://app.seermedical.com/studies/6dc191c3-45c0-4990-bdf0-8dc6060d1f6e) |           3 | Y  |        Y         |        N         |    N    |
| [MSEL_01575](https://app.seermedical.com/studies/3112c100-d71e-438f-96e5-4defc3b9cfa7)     | [5 days, 3 hr 14 min 20 s](https://app.seermedical.com/studies/3112c100-d71e-438f-96e5-4defc3b9cfa7)  |          82 | Y  |        N         |        N         |    N    |
| [MSEL_01870](https://app.seermedical.com/studies/d047df75-f5ea-4e67-8bde-40eae6125016)     | [4 days, 23 hr 45 min 38 s](https://app.seermedical.com/studies/d047df75-f5ea-4e67-8bde-40eae6125016) |           6 | Y  |        N         |        N         |    N    |
| [MSEL_00182](https://app.seermedical.com/studies/c5f9c2a1-81c5-404c-ae40-e55b606e4cd1)     | [3 days, 13 hr 15 min 1 s](https://app.seermedical.com/studies/c5f9c2a1-81c5-404c-ae40-e55b606e4cd1)  |           4 | Y  |        N         |        N         |    N    |
| [MSEL_01844](https://app.seermedical.com/studies/bab85744-4040-4496-b35e-dfb1ed638b73)     | [3 days, 17 hr 58 min 44 s](https://app.seermedical.com/studies/bab85744-4040-4496-b35e-dfb1ed638b73) |          42 | Y  |        N         |        N         |    N    |
| [MSEL_01842](https://app.seermedical.com/studies/72397e46-c424-4580-af50-1c460894664d)     | [3 days, 12 hr 14 min 44 s](https://app.seermedical.com/studies/72397e46-c424-4580-af50-1c460894664d) |          12 | Y  |        N         |        N         |    N    |
| [MSEL_01808](https://app.seermedical.com/studies/fba0f4f1-29c5-4387-b076-c6f788bbcaca)     | [3 days, 19 hr 37 min 28 s](https://app.seermedical.com/studies/fba0f4f1-29c5-4387-b076-c6f788bbcaca) |           9 | Y  |        N         |        N         |    N    |
| [MSEL_01860](https://app.seermedical.com/studies/446364b3-ccb3-4927-8a32-f70ab0f5aa13)     | [4 days, 53 min 18 s](https://app.seermedical.com/studies/446364b3-ccb3-4927-8a32-f70ab0f5aa13)       |           6 | N  |        Y         |        N         |    Y    |
| [MSEL_01859](https://app.seermedical.com/studies/fa0ddd6c-49e7-4fca-a2e6-6f93cd4f31e7)     | [21 hr 35 min 41 s](https://app.seermedical.com/studies/fa0ddd6c-49e7-4fca-a2e6-6f93cd4f31e7)         |          \- | N  |        N         |        N         |    Y    |
| [MSEL_01676](https://app.seermedical.com/studies/2cec788f-8577-4142-bd5e-696a72fcdcda)     | [7 days, 18 hr 34 min 53 s](https://app.seermedical.com/studies/2cec788f-8577-4142-bd5e-696a72fcdcda) |          11 | Y  |        N         |        Y         |    N    |
| [MSEL_01849](https://app.seermedical.com/studies/e41cb887-67a4-4f24-bdf3-a2b7fb94d67d)     | [5 days, 23 hr 51 min 9 s](https://app.seermedical.com/studies/e41cb887-67a4-4f24-bdf3-a2b7fb94d67d)  |           1 | Y  |        Y         |        N         |    N    |
| [MSEL_01843](https://app.seermedical.com/studies/069d4a4e-35cc-44cd-874a-d68d27d0a620)     | [4 days, 23 hr 21 min 47 s](https://app.seermedical.com/studies/069d4a4e-35cc-44cd-874a-d68d27d0a620) |           4 | Y  |        Y         |        Y         |    N    |
| [MSEL_00501](https://app.seermedical.com/studies/4f88cf00-8bf3-43fd-970d-9e434acf6edb)     | [3 days, 16 hr 9 min 29 s](https://app.seermedical.com/studies/4f88cf00-8bf3-43fd-970d-9e434acf6edb)  |          17 | Y  |        Y         |        N         |    N    |
| [MSEL_00172](https://app.seermedical.com/studies/92434006-5506-4d4a-97e4-f7232dfd68bd)     | [2 days, 21 hr 59 min 5 s](https://app.seermedical.com/studies/92434006-5506-4d4a-97e4-f7232dfd68bd)  |          12 | Y  |        Y         |        N         |    N    |
| [MSEL_01838](https://app.seermedical.com/studies/372482db-bcd8-4c79-85aa-16de76d8df69)     | [4 days, 17 hr 37 min 48 s](https://app.seermedical.com/studies/372482db-bcd8-4c79-85aa-16de76d8df69) |          11 | Y  |        N         |        Y         |    N    |
| [MSEL_01832](https://app.seermedical.com/studies/240cbd53-3c68-4358-b30b-52a50c55c6f0)     | [9 days, 17 hr 34 min 51 s](https://app.seermedical.com/studies/240cbd53-3c68-4358-b30b-52a50c55c6f0) |           4 | Y  |        Y         |        N         |    N    |
| [MSEL_01836](https://app.seermedical.com/studies/d4a2290e-b602-4ff3-9a07-b92e851b7c94)     | [2 days, 12 hr 47 min 37 s](https://app.seermedical.com/studies/d4a2290e-b602-4ff3-9a07-b92e851b7c94) |           4 | Y  |        Y         |        N         |    N    |
| [MSEL_01097](https://app.seermedical.com/studies/fff9aaa9-b104-46e8-9227-b1b76d6f333e)     | [4 days, 21 hr 5 min 12 s](https://app.seermedical.com/studies/fff9aaa9-b104-46e8-9227-b1b76d6f333e)  |           7 | Y  |        Y         |        N         |    N    |
| [MSEL_00502](https://app.seermedical.com/studies/19c473e1-14a3-4957-aab5-a5e0f59bc947)     | [2 days, 15 hr 19 min 20 s](https://app.seermedical.com/studies/19c473e1-14a3-4957-aab5-a5e0f59bc947) |           6 | Y  |        N         |        Y         |    N    |
| [MSEL_01763](https://app.seermedical.com/studies/5feefd8a-fdc5-4b9c-8f2e-cba554d0398e)     | [22 hr 16 min 33 s](https://app.seermedical.com/studies/5feefd8a-fdc5-4b9c-8f2e-cba554d0398e)         |          29 | Y  |        N         |        Y         |    N    |
| [MSEL_01550](https://app.seermedical.com/studies/1260d1ec-ba45-4360-8d64-99e5b8e2b3a4)     | [4 days, 21 hr 39 min 21 s](https://app.seermedical.com/studies/1260d1ec-ba45-4360-8d64-99e5b8e2b3a4) |           5 | Y  |        N         |        Y         |    N    |
| [MSEL_01853](https://app.seermedical.com/studies/82730685-2ac8-4be4-a297-1167008bc7d7)     | [1 days, 12 hr 23 min 9 s](https://app.seermedical.com/studies/82730685-2ac8-4be4-a297-1167008bc7d7)  |          13 | Y  |        Y         |        N         |    N    |


## Resources, code

Jupyter notebooks used for exploring the dataset structure and content, preprocessing possibilities:
* [explore_msg_data.ipynb](analysis/explore_msg_data.ipynb)  :   dataset content exploration and analysis  
* [preproc_data.ipynb](analysis/preproc_data.ipynb)       :   dataset preparation and preprocessing methods
* [explore_pp_data.ipynb](analysis/explore_pp_data.ipynb)  : exploration of preprocessed data

The solution consists of the following modules:  
* `main.py`     : the executable Main module containing the preparing, training, evaluation and prediction algorithms
* `datasets.py` : contains the custom datasets, data loaders and auxiliary methods related to data loading
* `msg_subject_data.py` : dataset helper and subject data manager class
* `model.py`    : contains the neural network model definitions
* `preproc_utils.py`    : functions for using and preprocessing the MSG dataset
* `utils.py`    : auxiliary and utility functions

## Configuration

Most configuration parameters are set in the `config.json` file.

| Category      | Configuration parameter              | Description                                                                       |
| ------------- | ------------------------------------ | --------------------------------------------------------------------------------- |
| environment   | device                               | preferred device to use (cpu, cuda, mps)                                          |
| environment   | project_root                         | root directory of the project files (data, models and results)                    |
| environment   | input_dir                            | subdirectory under data_root containing the raw MSG E4 data input                 |
| environment   | preproc_dir                          | subdirectory under data_root, where preprocessed data is saved                    |
| environment   | train_dir                            | subdirectory under data_root to store training files (tensorboard log and models) |
| environment   | results_dir                          | subdirectory under data_root, where forecasting results are saved                 |
| environment   | model_file                           | default model file name to store the best model                                   |
| training      | num_workers                          | number of workers in the dataloader                                               |
| training      | training_config/learning_rate        | learning rate to use during training                                              |
| training      | training_config/weight_decay         | weight decay for the Adam optimizer                                               |
| training      | training_config/evaluate_at          | step count to evaluate at                                                         |
| training      | training_config/early_stopping       | indicates if early stopping should be applied                                     |
| training      | training_config/early_stop_threshold | number of epochs without improvement that renders stopping                        |
| training      | training_config/scheduler_step       | step size for the LR scheduler (StepLR)                                           |
| training      | training_config/scheduler_gamma      | gamma value for the LR scheduler (StepLR)                                         |
| evaluation    | eval_threshold                       | default evaluation threshold for the binary classifier (e.g. 0.5)                 |
| evaluation    | eval_window                          | window size for evaluation (minutes)                                              |
| evaluation    | lead_seizure_separation              | separation of lead seizures in minutes                                            |
| evaluation    | preictal_setback                     | preictal setback time in minutes                                                  |
| evaluation    | interictal_separation                | interictal separation time in minutes                                             |
| evaluation    | target_labels                        | target label values to use                                                        |
| evaluation    | cross_validation                     | indicates if cross-validation should be used                                      |
| wearable | wearable_data/subjects               | list of subjects to process (command line overrides this setting)                 |
| wearable | wearable_data/freq                   | sampling frequency of the raw input data                                          |
| wearable | wearable_data/physio_band            | physiological narrowband limits (list of two values) for AccMag SQI calculation   |
| wearable | wearable_data/broad_band             | broadband limits (list of two values) for AccMag SQI calculation                  |
| wearable | wearable_data/no_overlap             | number of overlaping frames in spectral power SQI calculation                     |
| wearable | wearable_data/device                 | type of wearable device                                                           |
| wearable | wearable_data/device/sensor_channels | list of the channels of the raw input                                             |
| preprocessing | stft/freq                            | frequency for the Short-time Fourier Transform                                    |
| preprocessing | stft/seg_size                        | segment size for the Short-time Fourier Transform                                 |
| preprocessing | stft/overlap                         | overlap size for the Short-time Fourier Transform                                 |
| preprocessing | stft/freq_pool                       | max-pooling window size for frequency for the Short-time Fourier Transform        |
| preprocessing | stft/time_pool                       | max-pooling window size for time-domain in STFT                                   |
| preprocessing | stft/reduce                          | apply time-dimension reduction                                                    |
| dataset | dataset_config/train_test_split      | the train/test split ratio to use (from the dataset)                              |
| dataset | dataset_config/valid_ratio           | the validation/train ratio to use (from the train set)                            |
| dataset | dataset_config/finetune_ratio        | n/a                                                                               |
| dataset | dataset_config/scoring_set_size      | n/a                                                                               |
| dataset | dataset_config/reuse_scaler          | indicates if re-calculating scaler for z-scoring should be omitted                |
| dataset | dataset_config/exclude_threshold     | threshold for exclusion based on signal absence or low variability                |
| transforms | transforms_config/augment            | indicates if data augmentation should be applied                                  |
| transforms | transforms_config/type               | type of data augmentation                                                         |
| transforms | transforms_config/ratio              | max. number of copies to create                                                   |
| transforms | training_config/epochs               | number of training epochs (max.)                                                  |
| transforms | training_config/batch_size           | batch size to use during training                                                 |
| architecture | network_config/model_class           | the name of the modell class to use                                               |
| architecture | network_config/input_size            | the input size (feature input dimension for a single frame)                       |
| architecture | network_config/lstm_layers           | number of LSTM layers                                                             |
| architecture | network_config/hidden_units          | number of hidden units in the LSTM                                                |
| architecture | network_config/fc                    | linear layer input, output dimensions (list of two values)                        |
| architecture | network_config/lstm_dropout          | dropout rate between the LSTM layers                                              |
| architecture | network_config/bfc_dropout           | dropout rate before the fully-connected layer                                     |
| architecture | network_config/out_size              | output dimension                                                                  |


## Usage

### Preprocessing

`python3 -u main.py --mode preprocess --subjects <subject_ID>`

### Training

`python3 -u main.py --mode train --subjects <subject_ID>`

### Forecasting

`python3 -u main.py --mode forecast --subjects <subject_ID>`


