import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

class VGG16(nn.Module):
    """
    Neural network model based on VGG16 architecture.
    """

    def __init__(self, args: dict, device):
        """
        Define and allocate layers for this neural net.
        Args:
            num_classes (int): number of classes to predict with this model
        """
        super(VGG16, self).__init__()

        self.args = args
        self.num_classes = 1
        self.in_channels = self.args["num_channels"]

        # Define the VGG16 layers
        self.features = nn.Sequential(
            # Input: (1, 19, 800)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Conv1_1
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Conv1_2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool1
            # Output: (64, 9, 400)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Conv2_1
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Conv2_2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool2
            # Output: (128, 4, 200)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Conv3_1
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Conv3_2
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Conv3_3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool3
            # Output: (256, 2, 100)

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Conv4_1
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Conv4_2
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Conv4_3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # MaxPool4
            # Output: (512, 2, 50)

            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Conv5_1
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Conv5_2
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Conv5_3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # MaxPool5
            # Output: (512, 1, 25)
        )

        # Define the fully connected layers (classifier)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 25, 4096),  # FC1
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # FC2
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, self.num_classes),  # FC3 (output layer)
        )

        self.init_bias()  # Initialize weights and biases

    def init_bias(self):
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        #x = x.permute(0, 2, 1)
        x = torch.unsqueeze(x, dim=1)
        x = self.features(x)
        x = x.view(x.shape[0], -1)  # Flatten the output for the fully connected layers
        x = self.classifier(x)
        return x, 0

    def init_state(self, device):
        return 0
