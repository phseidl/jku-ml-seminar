import pandas as pd



if __name__ == "__main__":
    '''
    df1 = pd.read_csv('results/psd/results1.csv')
    df2 = pd.read_csv('results/psd/results2.csv')
    df3 = pd.read_csv('results/psd/results3.csv')
    df4 = pd.read_csv('results/psd/results4.csv')

    # merge the dataframes
    df_merged = pd.concat([df1, df2, df3, df4])
    # get mean by model
    df_merged_mean = df_merged.groupby(['Model']).mean().reset_index()
    # get variance by model
    df_merged_std = df_merged.groupby(['Model']).std().reset_index()

    mean = df_merged_mean.copy()
    std = df_merged_std.copy()

    # General metrics rounded to 2 decimals
    metrics_2dec = [
        "Test AUC", "Test AP", "Test TPR", "Test TNR",
        "Margin Onset Accuracy", "Margin Offset Accuracy"
    ]

    # Inference speed rounded to 6 decimals
    metrics_6dec = ["Average inference time"]


    # Format "mean ± std" for 2-decimal metrics
    def format_2dec(col):
        return mean[col].round(2).astype(str) + " ± " + std[col].round(2).astype(str)
    # Format for 6-decimal metric
    def format_6dec(col):
        return mean[col].round(4).astype(str) + " ± " + std[col].round(4).astype(str)


    # Build final DataFrame
    df_final = pd.DataFrame()
    df_final["Model"] = mean["Model"]
    df_final["AUROC"] = format_2dec("Test AUC")
    df_final["AUPRC"] = format_2dec("Test AP")
    df_final["TPR"] = format_2dec("Test TPR")
    df_final["TNR"] = format_2dec("Test TNR")

    # Combine Onset/Offset into a single margin column
    df_final["MARGIN (5sec) Acc (Onset/Offset)"] = (
            format_2dec("Margin Onset Accuracy") + " / " + format_2dec("Margin Offset Accuracy")
    )

    # Inference speed with 6-decimal formatting
    df_final["Inference Speed (sec)"] = format_6dec("Average inference time")

    # write df merged mean to latex table
    df_merged_mean.to_latex('results/psd/results_mean.tex', index=False, float_format="%.2f")

    # write df merged var to latex table
    df_merged_std.to_latex('results/psd/results_std.tex', index=False, float_format="%.2f")

    # write df combined to latex table
    df_final.to_latex('results/psd/results_combined.tex', index=False, float_format="%.2f")
'''
    df1 = pd.read_csv('results_MIT1.csv')
    df2 = pd.read_csv('results_MIT2.csv')
    df3 = pd.read_csv('results_MIT3.csv')
    # remove acc, precision, recall
    df1 = df1.drop(columns=['Test ACC', 'Precision', 'Recall'])
    df2 = df2.drop(columns=['Test ACC', 'Precision', 'Recall'])
    df3 = df3.drop(columns=['Test ACC', 'Precision', 'Recall'])


    # merge the dataframes
    df = pd.concat([df1, df2, df3])
    # get mean by model
    df_mean = df.groupby(['Model']).mean().reset_index()
    # get variance by model
    df_std = df.groupby(['Model']).std().reset_index()

    # Format "mean ± std" for 2-decimal metrics
    def format_2dec(col):
        return df_mean[col].round(2).astype(str) + " ± " + df_std[col].round(2).astype(str)
    # Format for 6-decimal metric
    def format_6dec(col):
        return df_mean[col].round(4).astype(str) + " ± " + df_std[col].round(4).astype(str)
    # Build final DataFrame
    df_final = pd.DataFrame()
    df_final["Model"] = df_mean["Model"]
    df_final["AUROC"] = format_2dec("Test AUC")
    df_final["AUPRC"] = format_2dec("Test AP")
    df_final["TPR"] = format_2dec("Test TPR")
    df_final["TNR"] = format_2dec("Test TNR")
    df_final["MARGIN (5sec) Acc (Onset/Offset)"] = (
            format_2dec("Margin Onset Accuracy") + " / " + format_2dec("Margin Offset Accuracy")
    )
    df_final["Inference Speed (sec)"] = format_6dec("Average inference time")

    # remove "_" in model names
    df_final["Model"] = df_final["Model"].str.replace("_", " ")
    df_mean["Model"] = df_mean["Model"].str.replace("_", " ")
    df_std["Model"] = df_std["Model"].str.replace("_", " ")
    # write df mean to latex table
    df_mean.to_latex('results_MIT_mean.tex', index=False, float_format="%.2f")
    # write df std to latex table
    df_std.to_latex('results_MIT_std.tex', index=False, float_format="%.2f")
    # write df final to latex table
    df_final.to_latex('results_MIT_combined.tex', index=False, float_format="%.2f")
