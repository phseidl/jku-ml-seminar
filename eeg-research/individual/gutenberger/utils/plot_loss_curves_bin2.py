import matplotlib.pyplot as plt 
import pandas as pd
import os

titles = ['Bin loss', 'Validation loss (MSE)']
models = ['CLEAN-B0', 'CLEAN-B1', 'CLEAN-B2', 'CLEAN-B3','CLEAN-B4']
loss_types = ['train', 'val']
# Define file paths

file_path = 'C:/Users/mgute/OneDrive/Dokumente/A_Universität/JKU/AI_Master/A_EEG/Master_Thesis_AI/wandb_training_curves/bin_loss'
output_dir = os.path.join(file_path, "plots")
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
save_name = 'bin_loss_curves'

# Create a 1x2 subplot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Loop through files and plot loss curves
for i, (loss_type, title) in enumerate(zip(loss_types, titles)):
    for model_name in models:
        file_dir = os.path.join(file_path, model_name+'_'+loss_type+'.csv')
        # Read CSV file
        df = pd.read_csv(file_dir)
        
        # Ensure correct column names
        epoch_column = df.columns[0] 

        # If title is "Validation Loss", add (step=0, loss=0.31)
        #if title == "Validation loss (MSE)":
            #df.loc[-1] = [0, 0.31, 0.31, 0.31]  # Add new row
            #df = df.sort_values(by=epoch_column).reset_index(drop=True)  # Ensure correct order

        # Plot loss curve
        alpha = 0.65 if loss_type == 'train' else 1
        axes[i].plot(df[epoch_column],  df.iloc[:, 1], label=model_name, alpha= alpha)
        axes[i].set_title(title, fontsize=20)
        axes[i].set_xlabel("Steps", fontsize=14)
        axes[i].set_ylabel("Loss", fontsize=14)
        axes[i].set_xlim((0, 80000))
        if loss_type == 'train':
            axes[i].set_ylim((3.4, 5.0)) 
        axes[i].grid(True)

        # Set y-axis limits based on title
        #if title == "Validation loss (MSE)":
            #axes[i].set_ylim((0.04, 0.15))

# Adjust layout and show/save plot
plt.tight_layout()
axes[-1].legend(models, fontsize = 15, loc='upper right', bbox_to_anchor=(1, 0.85))
save_path = os.path.join(output_dir, f"{save_name}.pdf")
plt.savefig(save_path, format="pdf", bbox_inches="tight")
plt.show()
