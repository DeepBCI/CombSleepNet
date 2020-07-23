import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        conv1 = nn.Conv2d(1, 10, (1, 200))
        pool1 = nn.MaxPool2d((1, 2))
        conv2 = nn.Conv2d(10, 20, (2, 32))
        conv3 = nn.Conv2d(20, 30, (2, 128))
        conv4 = nn.Conv2d(30, 40, (2, 512))
        pool2 = nn.MaxPool2d((1, 2))
        self.conv_module = nn.Sequential(conv1, nn.ReLU(), pool1, conv2, nn.ReLU(), conv3, nn.ReLU(), conv4, nn.ReLU(), pool2)

        fc1 = nn.Linear(14600, 100)
        fc2 = nn.Linear(100, 5)

        self.fc_module = nn.Sequential(fc1, nn.ReLU(), fc2)

        if use_cuda:
            self.conv_module = self.conv_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self, x, isfc):
        out = self.conv_module(x)
        dim = 1
        for d in out.size()[1:]:
            dim *= d

        if isfc:
            out = out.view(-1, dim)
            out = self.fc_module(out)
        else:
            out = out.permute(0, 3, 2, 1).reshape([-1, 200, 73])
        return out