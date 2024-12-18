# import numpy as np
# from pathlib import Path

# # Directory containing the .npy files
# input_dir = Path("/Volumes/PHILIPS/thesis/data/embeddings_pubchem10m")  # Replace with your directory
# output_file = input_dir / "combined_embeddings.npy"
# batch_size = 10  # Number of files to process in each batch

# # Get a sorted list of all .npy files
# npy_files = sorted(input_dir.glob("embeddings_pubchem10m.npy_*.npy"))

# # Initialize the combined embeddings file
# if output_file.exists():
#     combined_embeddings = np.load(output_file, mmap_mode='r+')
#     start_index = 0
# else:
#     combined_embeddings = None
#     start_index = 0

# # Process files in batches
# for i in range(start_index, len(npy_files), batch_size):
#     batch_files = npy_files[i:i + batch_size]
#     print(f"Processing batch {i // batch_size + 1}: {len(batch_files)} files")

#     # Load and concatenate the current batch
#     batch_embeddings = [np.load(f) for f in batch_files]
#     batch_combined = np.vstack(batch_embeddings)

#     if combined_embeddings is None:
#         # Initialize the combined embeddings with the first batch
#         combined_embeddings = batch_combined
#     else:
#         # Concatenate the new batch with the existing combined embeddings
#         combined_embeddings = np.vstack([combined_embeddings, batch_combined])

#     # Save the combined embeddings back to disk after each batch
#     np.save(output_file, combined_embeddings)
#     print(f"Saved combined embeddings after processing batch {i // batch_size + 1}")

# print("All files successfully combined into:", output_file)

import numpy as np
from pathlib import Path

# Directory containing the single .npy files
input_dir = Path("/Volumes/PHILIPS/thesis/data/embeddings_pubchem10m")  # Replace with your directory
combined_file = input_dir / "combined_embeddings.npy"

# Get a sorted list of all .npy files
input_dir_files = Path("/Volumes/PHILIPS/thesis/data/embeddings_pubchem10m/single embeddings")  # Replace with your directory
npy_files = sorted(input_dir_files.glob("embeddings_pubchem10m.npy_*.npy"))

# Load the combined embeddings
combined_embeddings = np.load(combined_file)

print(f"Number of individual files: {len(npy_files)}")
print(f"Shape of combined embeddings: {combined_embeddings.shape}")

# Check the order
current_index = 0
order_correct = True

for file in npy_files:
    # Load the embeddings from the individual file
    single_embeddings = np.load(file)
    single_length = single_embeddings.shape[0]
    
    # Check if the corresponding part in the combined embeddings matches
    if not np.array_equal(combined_embeddings[current_index:current_index + single_length], single_embeddings):
        print(f"Mismatch found in file: {file}")
        # print missmatch
        print(combined_embeddings[current_index:current_index + single_length])
        print(single_embeddings)
        order_correct = False
        break

    current_index += single_length

if order_correct:
    print("The order of embeddings in the combined file matches the individual files.")
else:
    print("The order of embeddings in the combined file does not match the individual files.")