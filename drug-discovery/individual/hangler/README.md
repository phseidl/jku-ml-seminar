# COATI-Model-Evaluation
This repository contains all necessary materials and code for replicating the experiments and analyses presented in my report on the [COATI model](https://github.com/terraytherapeutics/COATI). It includes detailed scripts and notebooks for training the model from scratch using the GuacaMol dataset, conducting linear probing tasks, and analyzing molecular generation capabilities.

The `coati` directory is from the original COATI repository with some minor changes in the `dataset.py` class. All other folders and files contain experiment and analysis of the COATI framework.

# Setup Installation

## Clone the Repository
Clone the repository to your local machine to get started with the setup:

```bash
git clone https://github.com/StefanHangler/COATI-Model-Evaluation.git
cd COATI-Model-Evaluation
```

## Option 1: Using Conda
If you do not have Conda installed, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual).

Create the Conda environment from the `environment.yml` file, which includes all necessary dependencies:

```bash
conda env create -f env.yml
```

Activate the Conda environment:

```bash
conda activate coati_model_eval
```

## Option 2: Using Virtual Environment
- For Unix/macOS systems:
  ```bash
  python3 -m venv env
  source env/bin/activate
  ```
- For Windows:
  ```bash
  python -m venv env
  .\env\Scripts\activate
  ```
Install the package using `pip` which will handle all dependencies automatically:

```bash
pip install .
```

Both methods will install the COATI-Model-Evaluation package and all required dependencies, setting up the environment for running or extending the application.

# Training on the GuacaMol Benchmark

The training script provided is designed to train a model on the GuacaMol benchmark using a specific configuration of the COATI framework. Below are detailed instructions and important considerations for using this script effectively.

## Downloading Necessary Files

**GuacaMol Dataset and Trained Models**: Download the GuacaMol dataset and pre-trained model checkpoints from this Google Drive link:
   - [GuacaMol Dataset and Models](https://drive.google.com/drive/folders/1s9KSn68cwuMsMuuAHlv___XMP014D9WX?usp=sharing)

   Place the downloaded files in the appropriate directories as follows:
   - Dataset: Place under `guacamol/data/`
   - Model Checkpoints: Place under `guacamol/train_model/model_checkpoints/`

## Configuration
Before running the training script, it is essential to configure the parameters according to your training requirements and hardware setup. Key parameters include:

- `args.gpus`: Number of GPUs available on your node. The script automatically adapts to the number of GPUs.
- `args.batch_size`: Batch size per GPU. Adjust according to your GPU memory capacity.
- `args.n_epochs`: Number of epochs to train the model. (in the report I did 7 but 25 would be much better for benchmark comparisons)
- `args.lr`: Learning rate for the optimizer.
- `args.data_dir`: Path to the directory where the GuacaMol dataset is stored.

These parameters can be modified directly within the script.

## Execution
To run the training script, navigate to the directory containing the script and execute:

```bash
python train_guacamol.py
```

The script sets up the training environment, initializes a distributed training session using PyTorch's multiprocessing capabilities, and starts the training process across the specified number of GPUs.

## Output
- **Model Checkpoints**: The trained model checkpoints are saved in the specified `args.model_dir` directory.
- **Logs**: Training logs and output are stored in the `args.output_dir` directory. These logs provide detailed information about the training progress and performance.

## Monitoring Training Progress
It is recommended to monitor the training progress through the logs to ensure that the training is proceeding as expected. Check for any errors or unexpected behavior, especially in the early stages of training.

## Post-Training
After training, you can evaluate the model using the evaluation scripts provided or integrate the model into your application. Ensure to properly handle the model checkpoints for inference or further training.

# Molecule Generation
This repository contains a Python script for generating molecular structures using various pretrained PyTorch models. The models are implemented following an abstract base class to ensure a consistent interface for molecule generation.

## Features
- Abstract base class for molecule generators to ensure consistency and reusability.
- Predefined classes for generating molecules with specific pretrained models.
- Easy extension to include new molecule generation models.

## Generating Molecules

### Run Script
Execute the main script to generate molecules:
```
python generate_molecules.py
```

This will generate molecules using all configured models and save them to CSV files in the current directory.

### Adding a New Model

To add a new model for molecule generation, follow these steps:

1. **Implement the New Model Class:**
   Create a new Python class that inherits from `MoleculeGenerator`. Implement all abstract methods:
   - `load_model()`: Method to load your model and tokenizer.
   - `generate_molecules()`: Method to generate molecules using the model.

   Example:
   ```python
   class NewModelGenerator(MoleculeGenerator):
       def __init__(self, device, doc_url):
           super().__init__(device)
           self.doc_url = doc_url
           self.model, self.tokenizer = self.load_model()

       def load_model(self):
           # Load your model and tokenizer here
           return model, tokenizer

       def generate_molecules(self, num_molecules=100):
           # Logic to generate molecules
           return generated_molecules
   ```

2. **Integrate the New Model into the Main Script:**
   Instantiate your model in the main script and call the `generate_molecules()` method.
   
   ```python
   new_model = NewModelGenerator(device, model_url)
   molecules = new_model.generate_molecules(100)
   new_model.save_molecules(molecules, "new_model_generated_molecules.csv")
   ```

## Evaluating Generated Molecules

The provided script offers comprehensive evaluation metrics for generated molecules, including validity, novelty, uniqueness, and the Fréchet ChemNet Distance (FCD) score. These metrics are crucial for assessing the quality of molecules generated by different models.

### Evaluation Metrics

- **Validity**: Measures the percentage of chemically valid molecules using RDKit.
- **Novelty**: Measures the proportion of generated molecules that are not found in the training set.
- **Uniqueness**: Assesses how unique the generated molecules are within the generated set.
- **FCD Score**: Compares the distribution of generated molecules to a reference set using a deep learning model to compute the FCD.

### Running the Evaluation

To evaluate generated molecules, follow these steps:

1. **Prepare your dataset**: Ensure your generated molecules are stored in a format compatible with the script, typically as `.csv` files containing SMILES strings.
2. **Specify the paths**: Modify the `path_list` dictionary in the script to point to your generated molecule files.
3. **Load training data**: Update the `training_smiles_path` variable to point to your training dataset. This dataset is used to compute novelty and as a reference for the FCD.
4. **Run the script**: Execute the script to calculate the evaluation metrics. Results are printed to the console and can be saved to a CSV file for further analysis.

### Example Usage

Here is an example of how to call the evaluation function for a set of generated molecules:

```python
from guacamol.evaluate_guacamol import evaluate_guacamol

generated_smiles = ["CCO", "Oc1ccccc1"]
training_smiles = ["CCO", "CC"]

evaluate_guacamol(
    generated_smiles=generated_smiles,
    training_smiles=training_smiles,
    model_name="ExampleModel"
)
```

This will output the validity, novelty, uniqueness, and FCD score for the `ExampleModel`. Adjust the `training_smiles` and `generated_smiles` variables as necessary to reflect your datasets.


# Linear Probing for ADMET Property Prediction

Linear probing is used to evaluate the quality of learned embeddings by training a simple model, such as logistic regression or random forest, on downstream tasks using these embeddings. This section explains how to use the linear probing utilities included in this repository to assess embeddings for various ADMET datasets.

## Downloading ADMET Datasets

Before you can run the linear probing tasks, it's essential to have the ADMET datasets available locally. This repository includes a utility function in `admet_dataset.py` that facilitates the downloading of these datasets from Terray's public S3 bucket.

### Using the Download Function

The function `download_admet_terray_data()` automates the process of downloading the datasets. Here's how you can use it:
**Execute the Download**: Run the following function to start downloading the datasets:

    ```python
    from admet_dataset import download_admet_terray_data
    download_admet_terray_data()
    ```

    This function checks for existing files before downloading to avoid unnecessary data transfer. If a dataset already exists locally, it will skip re-downloading it.

### What Happens Next?
After running the function, the datasets will be downloaded to the `./datasets` directory within your project structure. Each dataset's name is derived from the last segment of its URL, ensuring that they are stored with recognizable and consistent filenames.

### Troubleshooting
- **Download Issues**: If there are any issues during the download (e.g., due to network interruptions or permissions), the function will print an error message specifying the problem. Ensure that your AWS credentials are configured correctly if using `boto3`.
- **Manual Download**: If automated downloading fails, you may manually download the datasets from the provided links and place them in the `./datasets` directory.

## Setup

Before running the linear probing tasks, ensure that your ADMET datasets are prepared and that embeddings are updated accordingly. Use the provided script `update_dataset_embeddings` to preprocess the embeddings for your dataset.

## Relevant Scripts

- **`admet_dataset.py`**: Manages the loading and processing of ADMET datasets.
- **`train_scaffold_split.py`**: Contains utilities for splitting datasets based on molecular scaffolds, training models, and evaluating them using metrics such as AUROC, &Delta;AP (Delta Average Precision), RMSE, and Spearman correlation.

## Running Linear Probing

1. **Update ADMET Datasets**:
   Ensure that your ADMET datasets are prepared with necessary columns for linear probing. Use the `update_datasets` function to add `Metric` and `Task` columns to each dataset.
   
   ```python
   from admet_dataset import update_datasets
   update_datasets()
   ```

2. **Evaluate Models**:
   Use the `coati_linear_probing_eval` function from `train_scaffold_split.py` to perform linear probing. This function automatically handles dataset loading, model training, and evaluation.
   
   ```python
   from train_scaffold_split import coati_linear_probing_eval
   coati_linear_probing_eval()
   ```

3. **Results**:
   Evaluation metrics such as AUROC, &Delta;AP, RMSE, Spearman correlation, and their bootstrapped uncertainties will be printed during the execution. These results are also saved to a CSV file for further analysis.

## Note

For detailed information on each dataset and the expected model performance, refer to the `dataset_info` dictionary in `admet_dataset.py`. Adjust the base path to datasets or embeddings as necessary based on your project setup.

### Troubleshooting

- Ensure that all dataset paths and model URLs are correctly set in the scripts.


# License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.
