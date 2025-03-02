import torch
import torch.nn as nn

# Function to calculate soft bin probabilities for real-valued outputs
def compute_bin_probabilities(data, bins):
    """
    Compute the probability distribution over bins for real-valued data, clamping out-of-bounds values.
    
    Args:
        data (torch.Tensor): Real-valued model output, shape [batch_size, num_channels, signal_length].
        bins (torch.Tensor): Bin edges (1D tensor of size N+1 for N bins).
        
    Returns:
        torch.Tensor: Probability distribution over bins, shape [batch_size, num_channels, signal_length, num_bins].
    """
    # Clamp output to stay within bin range
    data = torch.clamp(data, min=bins[0], max=bins[-1])  # Clamp values within the bin range
    
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Compute bin centers
    bin_width = bins[1] - bins[0]  # Assume uniform bin widths

    # Compute Gaussian-like probabilities for each bin
    data = data.unsqueeze(-1)  # Add bin dimension
    bin_centers = bin_centers.to(data.device)  # Ensure same device
    distances = -((data - bin_centers) ** 2) / (2 * (bin_width ** 2))
    distances = torch.clamp(distances, min=-50, max=50)  # Clamp to avoid large values
    probabilities = torch.exp(distances)
    
    # Normalize probabilities along the bin dimension
    probabilities_sum = probabilities.sum(dim=-1, keepdim=True)
    probabilities = probabilities / (probabilities_sum + 1e-8)  # Add epsilon to avoid division by zero
    return probabilities

# Function to compute cross-entropy loss
def binned_cross_entropy_loss(output, target, bins):
    """
    Calculate cross-entropy loss for real-valued outputs with soft binning.
    
    Args:
        output (torch.Tensor): Model output, shape [batch_size, num_channels, signal_length].
        target (torch.Tensor): Ground truth, shape [batch_size, num_channels, signal_length].
        bins (torch.Tensor): Bin edges (1D tensor of size N+1 for N bins).
        
    Returns:
        torch.Tensor: Loss value.
    """
    # Compute bin indices for targets
    target_bins = bin_data(target, bins)  # Discrete bin indices
    num_bins = bins.size(0) - 1  # Number of bins
    
    # Compute soft probabilities for the output
    output_probs = compute_bin_probabilities(output, bins)
    
    # Flatten tensors for cross-entropy loss
    #batch_size, num_channels, signal_length = target_bins.shape
    target_bins_flat = target_bins.view(-1)
    output_probs_flat = output_probs.view(-1, num_bins)
    
    # Compute the cross-entropy loss
    loss = nn.CrossEntropyLoss()(output_probs_flat, target_bins_flat)
    return loss

# Function to bin data into discrete indices
def bin_data(data, bins):
    """
    Bin the data based on specified bins.
    
    Args:
        data (torch.Tensor): Input tensor of shape [batch_size, num_channels, signal_length].
        bins (torch.Tensor): Bin edges (1D tensor of size N+1 for N bins).

    Returns:
        torch.Tensor: Tensor of bin indices for the data, shape [batch_size, num_channels, signal_length].
    """
    bin_indices = torch.bucketize(data, bins, right=False) - 1
    bin_indices = torch.clamp(bin_indices, min=0, max=bins.size(0) - 2)  # Ensure indices are valid
    return bin_indices

# Example usage
if __name__ == "__main__":
    # Parameters
    batch_size = 8
    num_channels = 18
    signal_length = 512
    bin_size = 0.1  # Example bin size
    bin_edges = torch.arange(-1.0, 1.0 + bin_size, bin_size)  # Define bin edges
    
    # Simulate example data
    target = torch.rand(batch_size, num_channels, signal_length) * 2 - 1  # Target in range [-1, 1]
    output = torch.randn(batch_size, num_channels, signal_length)  # Real-valued model output
    
    # Compute loss
    loss = binned_cross_entropy_loss(output, target, bin_edges)
    print(f"Loss: {loss.item()}")
