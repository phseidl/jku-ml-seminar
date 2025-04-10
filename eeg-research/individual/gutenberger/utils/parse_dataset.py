import os
import re

# Replace this with the path to your folder
folder_path = 'C:/Users/mgute/OneDrive/Dokumente/A_Universität/JKU/AI_Master/A_EEG/Practical Work in AI/CLEEGN/CLEEGN_TUH/torch-CLEEGN/data/TUH_TUSZ/TUH_dataset_PROCESSED_new/original'

# Get all files in the folder
files = os.listdir(folder_path)

# Use a regular expression to extract numbers from filenames
numbers = [re.search(r'\d+', file).group() for file in files if file.endswith('.set')]

# Print the list of numbers
for number in numbers:
    print(f'    - ["{number}"]')