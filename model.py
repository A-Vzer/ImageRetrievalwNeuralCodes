import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class NeuralCodes(nn.Module):
    def __init__(self, code_length=2048, classes=190):
        super().__init__()
        self.flatten = Flatten()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=4)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=288, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=288, out_channels=288, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=288, out_channels=256, kernel_size=3)
        self.dense1 = nn.Linear(in_features=1024, out_features=code_length)
        self.dense2 = nn.Linear(in_features=code_length, out_features=code_length)
        self.dense3 = nn.Linear(in_features=code_length, out_features=classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=5, stride=2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(F.max_pool2d(self.conv5(x), kernel_size=3, stride=2))
        x = self.flatten(x)
        x = F.relu(self.dense1(F.dropout(x)))
        x = F.relu(self.dense2(F.dropout(x)))
        x = F.log_softmax(self.dense3(F.dropout(x)))

        return x

    def get_codes(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=5, stride=2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(F.max_pool2d(self.conv5(x), kernel_size=3, stride=2))
        code1 = self.flatten(x)
        code2 = F.relu(self.dense1(F.dropout(code1)))
        code3 = F.relu(self.dense2(F.dropout(code2)))
        return code1, code2, code3