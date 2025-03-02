import pandas as pd

def statistics_to_csv(input_csv, output_csv):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(input_csv)

    # Initialize an empty DataFrame for statistics
    stats = pd.DataFrame()

    # Calculate statistics for each column
    stats['Column'] = data.columns
    stats['Min'] = data.min().values
    stats['Max'] = data.max().values
    stats['Mean'] = data.mean().values
    stats['Std'] = data.std().values

    # Save the statistics DataFrame to a new CSV file
    stats.to_csv(output_csv, index=False)

    print(f"Column statistics saved to '{output_csv}'")


model = 'upt4eeg'
input_csv = "/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/eval_metrics_" + model + ".csv"  
output_csv = "/system/user/studentwork/gutenber/logs/TUH/UPT4EEG/eval_statistics_" + model + ".csv"
statistics_to_csv(input_csv, output_csv)