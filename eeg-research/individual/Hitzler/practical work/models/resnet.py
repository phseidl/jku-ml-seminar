import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, is_psd, stride=1):
        super(BasicBlock, self).__init__()

        kernel_size = (1, 7) if is_psd else (1, 9)
        padding = (0, 3) if is_psd else (0, 4)
        
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=(1, stride), padding=padding, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=(1, 1), padding=padding, bias=False),
            nn.BatchNorm2d(planes)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=(1, stride), bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.net(x) + self.shortcut(x)
        return F.relu(out)


class RESNET18_CONV2D(nn.Module):
    def __init__(self, args: dict, device):
        super(RESNET18_CONV2D, self).__init__()
        
        self.enc_model = args["enc_model"]
        num_blocks = [2, 2, 2, 2]
        block = BasicBlock
        self.in_planes = 64
        self.in_channels = args["num_channels"]

        self.features = self.enc_model != "raw"
        self.is_psd = self.enc_model in ["psd1", "psd2"]
        
        if self.is_psd:
            self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(1, 15), stride=(1, 2), padding=(0, 7), bias=False)
        elif self.enc_model == "raw":
            self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 51), stride=(1, 4), padding=(0, 25), bias=False)
        elif self.enc_model == "sincnet":
            self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 21), stride=(7, 2), padding=(0, 10), bias=False)
        else:
            self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], self.is_psd, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], self.is_psd, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], self.is_psd, stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], self.is_psd, stride=2)
        self.fc1 = nn.Linear(4 * 256 * block.expansion, 1)

        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, is_psd, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, is_psd, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.enc_model != "raw":
            x = self.feature_extractor[self.enc_model](x)
        else:
            x = torch.unsqueeze(x, dim=1)
        
        x = self.conv1(x)
        out = F.relu(self.bn1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.AdaptiveAvgPool2d((1, 4))(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out, 0

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def init_state(self, device):
        return 0
