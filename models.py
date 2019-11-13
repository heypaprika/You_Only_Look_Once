import shutil

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torchsummary.torchsummary import summary
from utilities import dataloader
from utilities.dataloader import VOC
from utilities.utils import one_hot
import numpy as np
import matplotlib.pyplot as plt


class YOLOv1(nn.Module):
    def __init__(self, args):
        super(YOLOv1, self).__init__()
        self.dropout = args.dropout
        self.num_classes = args.num_class
        self.momentum = 0.01
        self.Pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # layer1
        self.Conv_7x7_3_64_s2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        
        self.Conv_3x3_64_192 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192, momentum=self.momentum),
            nn.LeakyReLU()
        )
        
        self.Conv_1x1_192_128 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128, momentum=self.momentum),
            nn.LeakyReLU()
        )
        self.Conv_3x3_128_256 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=self.momentum),
            nn.LeakyReLU()
        )
        self.Conv_1x1_256_256 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256, momentum=self.momentum),
            nn.LeakyReLU()
        )
        self.Conv_3x3_256_512 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=self.momentum),
            nn.LeakyReLU(),
        )
        
        self.Conv_1x1_512_256 = nn.Sequential(
            nn.Conv2d(512,256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256, momentum=self.momentum),
            nn.LeakyReLU()
        )
        self.Conv_1x1_512_512 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512, momentum=self.momentum),
            nn.LeakyReLU()
        )
        self.Conv_3x3_512_1024 = nn.Sequential(
            nn.Conv2d(512,1024,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=self.momentum),
            nn.LeakyReLU()
        )
        
        self.Conv_1x1_1024_512 = nn.Sequential(
            nn.Conv2d(1024,512,kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512, momentum=self.momentum),
            nn.LeakyReLU()
        )
        self.Conv_3x3_1024_1024_s2 = nn.Sequential(
            nn.Conv2d(1024,1024,kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024, momentum=self.momentum),
            nn.LeakyReLU()
        )
        
        self.Conv_3x3_1024_1024 = nn.Sequential(
            nn.Conv2d(1024,1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=self.momentum),
            nn.LeakyReLU()
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 7 * 7 * (5 + self.num_classes))
        )
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.feature = nn.Sequential(
            self.Conv_7x7_3_64_s2(),
            self.Pool(),
            self.Conv_3x3_64_192(),
            self.Pool(),
            self.Conv_1x1_192_128(),
            self.Conv_3x3_128_256(),
            self.Conv_1x1_256_256(),
            self.Conv_3x3_256_512(),
            self.Pool(),
            self.Conv_1x1_512_256(),
            self.Conv_3x3_256_512(),
            self.Conv_1x1_512_256(),
            self.Conv_3x3_256_512(),
            self.Conv_1x1_512_256(),
            self.Conv_3x3_256_512(),
            self.Conv_1x1_512_256(),
            self.Conv_3x3_256_512(),
            self.Conv_1x1_512_512(),
            self.Conv_3x3_512_1024(),
            self.Pool(),
            self.Conv_1x1_1024_512(),
            self.Conv_3x3_512_1024(),
            self.Conv_1x1_1024_512(),
            self.Conv_3x3_512_1024(),
            self.Conv_3x3_1024_1024(),
            self.Conv_3x3_1024_1024_s2(),
            self.Conv_3x3_1024_1024(),
            self.Conv_3x3_1024_1024()
        )

        self.FC = nn.Sequential(
            self.fc1(),
            self.fc2()
        )

    def forward(self, x):
        out = self.feature(x)
        out = out.reshape(out.size(0), -1)
        out = self.FC(out)
        out = out.reshape((-1, 7, 7, (5 + self.num_classes)))
        out[:,:,:,0] = torch.sigmoid(out[:,:,:,0])
        out[:,:,:,5:] = torch.sigmoid(out[:,:,:,5:])
        return out


def detection_loss_4_yolo(pred, target, l_coord, l_noobj, device):
    b = target.size(0)
    n = pred.size(-1)

    objness1_pred = pred[:,:,:,0]
    x_offset1_pred = pred[:,:,:,1]
    y_offset1_pred = pred[:,:,:,2]
    w_ratio1_pred = pred[:,:,:,3]
    h_ratio1_pred = pred[:,:,:,4]
    class_pred = pred[:,:,:,5:]

    num_class = class_pred.size(-1)

    objness_label = target[:,:,:,0]
    x_offset_label = target[:,:,:,1]
    y_offset_label = target[:,:,:,2]
    w_ratio_label = target[:,:,:,3]
    h_ratio_label = target[:,:,:,4]
    class_label = one_hot(class_pred, target[:,:,:,5], device)

    noobjness_label = torch.neg(torch.add(objness_label, -1))

    obj_coord1_loss = l_coord * torch.sum(
        objness_label
        *
        (
            torch.pow(x_offset1_pred - x_offset_label, 2)
            +
            torch.pow(y_offset1_pred - y_offset_label, 2)
        )
    )

    obj_size1_loss = l_coord * torch.sum(
        objness_label
        *
        (
            torch.pow(w_ratio1_pred - torch.sqrt(w_ratio_label), 2)
            +
            torch.pow(h_ratio1_pred - torch.sqrt(h_ratio_label), 2)
        )
    )

    objness1_loss = torch.sum(
        objness_label
        *
        torch.pow(
            objness1_pred - objness_label,
            2
        )
    )

    noobjness1_loss = l_noobj * torch.sum(
        noobjness_label
        *
        torch.pow(
            objness1_pred - objness_label,
            2
        )
    )

    objectness_cls_map = objness_label.unsqueeze(-1)
    for i in range(num_class - 1):
        objectness_cls_map = torch.cat((objectness_cls_map, objness_label.unsqueeze(-1)), 3)

    obj_class_loss = torch.sum(
        objectness_cls_map
        *
        torch.pow(
            class_pred - class_label,
            2
        )
    )

    total_loss = obj_coord1_loss + obj_size1_loss + objness1_loss + noobjness1_loss + obj_class_loss
    total_loss = total_loss / b

    return total_loss, [obj_coord1_loss/b, obj_size1_loss/b, objness1_loss/b, noobjness1_loss/b, obj_class_loss/b]
