import torch
import torch.nn as nn
from torchvision import models

#__all__ = ['featureNet', 'ResNet50']

class Densenet121(nn.Module):

    def __init__(self, pretrained=True):
        super(Densenet121, self).__init__()

        self.base_model = models.densenet121(pretrained=pretrained)
        self.base_model = self.base_model._modules['features']
        ## base_model

        self.global_pool = nn.AvgPool2d(7, stride=1)
        self.last_linear = nn.Linear(1024, 1000)


    def features(self, x):
        x = self.base_model(x)
        return x


    def logits(self, features):
        x = self.global_pool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x


    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

if __name__ == '__main__':
    net = Densenet121()