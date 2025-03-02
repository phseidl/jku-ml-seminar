import matplotlib.pyplot as plt
import pandas as pd
import os

titles = ['Ensemble loss', 'Validation loss (MSE)', 'MSE loss', 'Bin loss']
# Define file paths
models = ['CLEAN-E1', 'CLEAN-E2', 'CLEAN-E3', 'CLEAN-E4']
loss_types = ['ensemble', 'val', 'mse', 'bin']
file_path = 'C:/Users/mgute/OneDrive/Dokumente/A_Universität/JKU/AI_Master/A_EEG/Master_Thesis_AI/wandb_training_curves/ensemble_loss'
output_dir = os.path.join(file_path, "plots")
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
save_name = 'ensemble_loss_curves'

# Create a 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Flatten axes array for easy iteration
axes = axes.flatten()

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
        alpha = 1.0 if loss_type=='val' else 0.6
        axes[i].plot(df[epoch_column],  df.iloc[:, 1], label=model_name, alpha = alpha)  #, color='b'
        axes[i].set_title(title, fontsize = 20)
        axes[i].set_xlabel("Steps", fontsize = 14)
        axes[i].set_ylabel("Loss", fontsize = 14)
        #axes[i].legend()
        axes[i].grid(True)
        #if title == "Validation loss (MSE)":
            #axes[i].set_ylim((0.04, 0.15))
        #if title == "Ensemble loss":
            #axes[i].set_ylim((0.38, 0.8))
        #if title == "MSE loss":
            #axes[i].set_ylim((0.0, 0.4))

# Adjust layout and show plot
plt.tight_layout()
axes[1].legend(models, fontsize = 15, loc='upper right')
save_path = os.path.join(output_dir, f"{save_name}.pdf")
plt.savefig(save_path, format="pdf", bbox_inches="tight")
plt.show()
