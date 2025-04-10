import numpy as np

def group_data_per_chan(output_pos, y, t):
    positions = np.unique(output_pos, axis = 0)    #20 unique 6d positions
    #print(positions.shape)
    channel_data = {tuple(pos): [] for pos in positions}
    channel_time = {tuple(pos): [] for pos in positions}
    
    # Group the data points based on their channel position
    for i in range(len(y)):
        channel_pos = tuple(output_pos[i])  # Extract the position as a tuple
        channel_data[channel_pos].append(np.array(y[i].cpu()))
        channel_time[channel_pos].append(np.array(t[i].cpu()))
    return channel_data, channel_time