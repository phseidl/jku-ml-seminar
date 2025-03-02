import pandas as pd
import os

# List of CSV filenames to process
file_path = 'C:/Users/mgute/OneDrive/Dokumente/A_Universität/JKU/AI_Master/A_EEG/Master_Thesis_AI/gpu_server/eval'
filenames = ['metrics_summary_upt4eeg_small_tuh_same_montage_random.csv', 
             'metrics_summary_upt4eeg_small_rdm_cd_same_montage_random.csv', 
             'metrics_summary_upt4eeg_small_rdm_cd_montage_random.csv', 
             'metrics_summary_upt4eeg_small_rdm_montage_random.csv', 
             'metrics_summary_upt4eeg_CLEAN-B0_montage_random.csv',
             'metrics_summary_upt4eeg_CLEAN-B1_montage_random.csv',
             'metrics_summary_upt4eeg_CLEAN-B2_montage_random.csv',
             'metrics_summary_upt4eeg_CLEAN-B3_montage_random.csv',
             'metrics_summary_upt4eeg_CLEAN-B4_montage_random.csv',
             'metrics_summary_upt4eeg_CLEAN-E1_montage_random.csv',
             'metrics_summary_upt4eeg_CLEAN-E2_montage_random.csv',
             'metrics_summary_upt4eeg_CLEAN-E3_montage_random.csv',
             'metrics_summary_upt4eeg_CLEAN-E4_montage_random.csv',] 
row_names = ['CLEAN0', 
             'CLEAN1', 
             'CLEAN2', 
             'CLEAN3', 
             'CLEAN-B0',
             'CLEAN-B1',
             'CLEAN-B2',
             'CLEAN-B3', 
             'CLEAN-B4',
             'CLEAN-E1',
             'CLEAN-E2',
             'CLEAN-E3',
             'CLEAN-E4'] 

save_name = 'latex_table_rand.txt'
table_label = 'performance_comp_hyperparams_rand'
caption = '{Comparison of CLEAN model performance across all test subjects using randomly selected bipolar channel input.}'

# Initialize an empty list to store the rows for the LaTeX table
table_data = []

# Define the desired metric order for the LaTeX table
desired_order = ['MSE', 'SNR', 'PCC', 'R^2']

# Process each file
for filename in filenames:
    # Read the CSV file
    df = pd.read_csv(os.path.join(file_path, filename))
    
    # Create a dictionary to store the metric values
    metric_dict = {metric: (mean, std) for metric, mean, std in zip(df['Metric'], df['Mean'], df['Standard Deviation'])}
    
    # Reorder the metrics according to the desired order
    row = [f'${metric_dict[metric][0]:.3f} \\pm {metric_dict[metric][1]:.3f}$' for metric in desired_order]
    
    # Append the row (corresponding to the current filename) to the table data
    table_data.append(row)

# Create the LaTeX table header
latex_table = "\\begin{table}[!htb]\n"
latex_table += "\\centering\n"
latex_table += "\\small\n"
latex_table += "\\begin{tabular}{l|cccc}\n"
latex_table += "\\textbf{} & \\textbf{MSE} $\\downarrow$ & \\textbf{SNR_{dB}} $\\uparrow$ & \\textbf{PCC} $\\uparrow$ & $\\mathbf{R^2} \\uparrow$ \\\\ \\hline\n"

# Add the rows for each filename
for i, row_name in enumerate(row_names):
    latex_table += f"{row_name} & " + " & ".join(table_data[i]) + " \\\\ \n"

# Close the table
latex_table += "\\end{tabular}\n"
latex_table += "\\caption"+ caption + "\n"
latex_table += "\\label{tab:" + table_label + "}\n"
latex_table += "\\end{table}"


with open(os.path.join(file_path, save_name), 'w') as f:
    f.write(latex_table)

# Output the LaTeX table
print(latex_table)
