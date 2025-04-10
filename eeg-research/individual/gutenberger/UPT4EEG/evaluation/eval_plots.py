import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# List of CSV files (replace with your file paths)
csv_files = ["subject_metrics_summary_CLEEGN.csv", "subject_metrics_summary_IC_U_Net.csv", "subject_metrics_summary_OneD_Res_CNN.csv", "subject_metrics_summary_upt4eeg_small_tuh.csv"]
model_names = ['CLEEGN', 'IC-U-Net', '1D-ResCNN', 'CLEAN']
path = "UPT4EEG/evaluation/tables"
save_path = "UPT4EEG/evaluation/plots" 

# Function to extract model name from the filename
def extract_model_name(filename):
    return filename.split("summary_")[1].split(".csv")[0]

# Read CSV files into a dictionary of DataFrames
model_data = {model_name: pd.read_csv(os.path.join(path, file), index_col=0) for file, model_name in zip(csv_files, model_names)}

# Get the list of metrics (rows) and subjects (columns) from the first model
metrics = model_data[next(iter(model_data))].index
subjects = model_data[next(iter(model_data))].columns

# Number of models
num_models = len(model_data)

# Plot each metric as a grouped bar chart
for metric in metrics:
    plt.figure(figsize=(12, 6))
    
    # Bar width and positions
    bar_width = 0.2
    x = np.arange(len(subjects))  # Positions for subjects
    
    # Plot bars for each model
    for i, (model_name, data) in enumerate(model_data.items()):
        plt.bar(x + i * bar_width, data.loc[metric], bar_width, label=model_name)
    
    # Customize the plot
    #plt.title(f'Comparison of {metric} Across Models', fontsize=14)
    metric_label = metric if metric != 'SNR' else 'SNR in dB'
    plt.xlabel('Subjects', fontsize=16)
    plt.ylabel(metric_label, fontsize=20)
    subject_labels = [str(int(subject)) for subject in subjects]
    plt.xticks(x + bar_width * (num_models - 1) / 2, subject_labels, rotation=45, fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=15) if metric != 'MSE' else plt.legend(fontsize=15, loc="upper center", bbox_to_anchor=(0.75, 1.0))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot as a PDF
    pdf_path = os.path.join(save_path, f'{metric}_comparison.pdf')
    plt.tight_layout()
    plt.savefig(pdf_path, format='pdf')
    plt.close()  # Close the figure to free up memory

print(f"Plots saved in the directory: {save_path}")