import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = 'C:/Users/mgute/OneDrive/Dokumente/A_Universität/JKU/AI_Master/A_EEG/Master_Thesis_AI/wandb_training_curves/baseline_models_al6td3wi'
output_dir = os.path.join(file_path, "plots")
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
save_name = 'baseline_loss_curves'

# Paths to your CSV files
train_csv_file = os.path.join(file_path, 'wandb_export_2025-02-15T17_46_40.379+01_00_train.csv')  # Replace with the actual path for the training loss CSV
val_csv_file = os.path.join(file_path, 'wandb_export_2025-02-15T17_46_25.093+01_00_val.csv')   # Replace with the actual path for the validation loss CSV

# Read the training and validation loss data
train_data = pd.read_csv(train_csv_file)
val_data = pd.read_csv(val_csv_file)

# Extract training and validation loss for each model from the respective files
# For the first model, use columns 0, 4, 7, etc. and for validation, same pattern
epochs = train_data.iloc[:, [0]] + 1
cleegn_train = train_data.iloc[:, [1]]  # Columns 0, 4, 7 for CLEEGN
ic_unet_train = train_data.iloc[:, [4]]  # Columns 1, 5, 8 for IC-U-Net
rescnn_train = train_data.iloc[:, [7]]  # Columns 2, 6, 9 for 1D-ResCNN

cleegn_val = val_data.iloc[:, [1]]  # Columns 0, 4, 7 for CLEEGN
ic_unet_val = val_data.iloc[:, [4]]  # Columns 1, 5, 8 for IC-U-Net
rescnn_val = val_data.iloc[:, [7]]  # Columns 2, 6, 9 for 1D-ResCNN

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training loss subplot (left)
axes[0].plot(epochs, cleegn_train, label='CLEEGN')
axes[0].plot(epochs, ic_unet_train, label='IC-U-Net')
axes[0].plot(epochs, rescnn_train, label='1D-ResCNN')
axes[0].set_title('Training Loss', fontsize=20)
axes[0].set_xlabel('Epochs', fontsize=14)
axes[0].set_ylabel('MSE', fontsize=14)
axes[0].set_xlim((1,80))
axes[0].legend(fontsize=15)

# Validation loss subplot (right)
axes[1].plot(epochs, cleegn_val, label='CLEEGN')
axes[1].plot(epochs, ic_unet_val, label='IC-U-Net')
axes[1].plot(epochs, rescnn_val, label='1D-ResCNN')
axes[1].set_title('Validation loss', fontsize=20)
axes[1].set_xlabel('Epochs', fontsize=14)
axes[1].set_ylabel('MSE', fontsize=14)
axes[1].set_xlim((1,80))
axes[1].legend(fontsize=15)

# Adjust layout
plt.tight_layout()

save_path = os.path.join(output_dir, f"{save_name}.pdf")
plt.savefig(save_path, format="pdf", bbox_inches="tight")

# Show the plot
plt.show()
