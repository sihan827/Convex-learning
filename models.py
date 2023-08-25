import torch.nn as nn
from collections import OrderedDict


class BasicClassifier(nn.Module):

    def __init__(self, img_size=32, img_channel=1, feature=2, bottleneck_pos=2):
        super(BasicClassifier, self).__init__()

        layer0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=img_channel, out_channels=feature,
             kernel_size=3, stride=1, padding=1, bias=True)),
            ('bn1', nn.BatchNorm2d(num_features=feature)),
            ('actv1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=4, stride=4)),
            ('flatten', nn.Flatten()),
        ]))

    def forward(self, x):
        x = self.front_network(x)
        x = self.back_network(x)
        return x
    

class LinearClassifier(nn.Module):

    def __init__(self, img_size=32, z_dim=4, img_channel=1):
        super(LinearClassifier, self).__init__()
        self.z_dim = z_dim

        front_layer = nn.Sequential(OrderedDict([
            ('flat', nn.Flatten()),
            ('fc1', nn.Linear(int(img_size ** 2 * img_channel), self.z_dim, bias=True)),
            ('actv1', nn.ReLU(inplace=True)),
        ]))

        back_layer = nn.Sequential(OrderedDict([
            ('fc2', nn.Linear(self.z_dim, 10, bias=True)),

        ]))

        self.front_network = front_layer
        self.back_network = back_layer

    def forward(self, x):
        x = self.front_network(x)
        x = self.back_network(x)
        return x