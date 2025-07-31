import torch
import torch.nn as nn


# Code reference: https://github.com/dansuh17/alexnet-pytorch

class AlexNet(nn.Module):
    """
    Neural network model consisting of layers proposed by AlexNet paper.
    """

    def __init__(self, args: dict, device):
        super(AlexNet, self).__init__()

        self.args = args
        self.features = True

        self.net = nn.Sequential(
            nn.ReLU(),
            nn.LocalResponseNorm(size=25, alpha=0.0001, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Conv2d(64, 128, (1, 15), padding=(0, 7)),
            nn.ReLU(),
            nn.LocalResponseNorm(size=25, alpha=0.0001, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Conv2d(128, 256, (1, 15), padding=(0, 7)),
            nn.ReLU(),
            nn.Conv2d(256, 128, (1, 15), padding=(0, 7)),
            nn.ReLU(),
            nn.Conv2d(128, 64, (1, 15), padding=(0, 7)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )
        self.features = False
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 51), stride=(1, 4))
        self.output_size = 20 * 5
        in_feature = 64 * self.output_size
        self.fc1 = nn.Linear(in_features=64 * self.output_size, out_features=in_feature // 2)

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.args["dropout"]),
            nn.Linear(in_feature // 2, in_feature // 2),
            nn.ReLU(),
            nn.Linear(in_feature // 2, 1),
        )

        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.net[3].bias, 1)
        nn.init.constant_(self.net[9].bias, 1)
        nn.init.constant_(self.net[11].bias, 1)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.classifier(x)

        return x, 0

    def init_state(self):
        return 0
