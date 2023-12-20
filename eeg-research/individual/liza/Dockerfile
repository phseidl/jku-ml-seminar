# Start from the PyTorch image with CUDA and cuDNN
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your script
COPY . .

# Run your script
CMD ["torchrun", "run_multiclass_supervised.py", "--dataset", "TUEV", "--in_channels", "16", "--n_classes", "6", "--sampling_rate", "200", "--token_size", "200", "--hop_length", "100", "--sample_length", "5", "--batch_size", "128", "--model", "BIOT"]