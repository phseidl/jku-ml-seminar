import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


val_files = [
    'Jan23_19-21-01/wandb_export_2025-01-29T16_28_15.572+01_00_val.csv',
    'Jan23_19-11-27/wandb_export_2025-01-29T16_36_49.405+01_00_val.csv', 
    'Jan23_19-12-03/wandb_export_2025-01-29T16_35_09.113+01_00_val.csv',
    'Jan23_19-12-29/wandb_export_2025-01-29T16_30_29.288+01_00_val.csv',
    ]

train_files = [
    'Jan23_19-21-01/wandb_export_2025-01-29T16_28_23.873+01_00_train.csv',
    'Jan23_19-11-27/wandb_export_2025-01-29T16_36_53.784+01_00_train.csv', 
    'Jan23_19-12-03/wandb_export_2025-01-29T16_35_14.254+01_00_train.csv',
    'Jan23_19-12-29/wandb_export_2025-01-29T16_30_34.891+01_00_train.csv',
    ]


# Load the data from the CSV file
file_path = 'C:/Users/mgute/OneDrive/Dokumente/A_Universität/JKU/AI_Master/A_EEG/Master_Thesis_AI/wandb_training_curves'

# Output directory for plots
output_dir = os.path.join(file_path, "plots")
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Function to load and plot loss curves
def plot_loss_curves(files, title, save_name, ylim_upper=None, alpha=None):
    alpha = 1.0 if alpha is None else alpha
    plt.figure(figsize=(8, 5))
    
    for i, file in enumerate(files):
        df = pd.read_csv(os.path.join(file_path, file))
        
        # Assuming the loss column name is 'loss' and the epoch is in the first column
        epoch_column = df.columns[0]  # Typically the first column is the step/epoch

        plt.plot(df[epoch_column], df.iloc[:, 1], alpha=alpha, label = f'CLEAN {i}')  # Use filename part as legend

    plt.xlabel("Steps", fontsize = 18)
    plt.ylabel("Loss", fontsize = 18)
    #plt.title(title)
    plt.legend(loc='upper right', fontsize = 14)
    plt.grid(True)
    if ylim_upper is not None:
        plt.ylim((0, ylim_upper))
    plt.xlim((0, df[epoch_column].max()))

    #plt.show()

    # Save the plot as a PDF
    save_path = os.path.join(output_dir, f"{save_name}.pdf")
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()  # Close the figure to free memory

# Plot and save validation loss
plot_loss_curves(val_files, "Validation Loss Curves", "validation_loss")

# Plot and save training loss
plot_loss_curves(train_files, "Training Loss Curves", "training_loss", ylim_upper=0.4, alpha = 0.7)

print(f"Plots saved in: {output_dir}")