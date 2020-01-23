import shutil

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import net

from sklearn.metrics.pairwise import euclidean_distances

from torchsummary.torchsummary import summary
from utilities import dataloader
from utilities.dataloader import VOC
from utilities.utils import one_hot
import numpy as np
import matplotlib.pyplot as plt

class End2End(nn.Module):
    def __init__(self, num_class, dropout):
        super(End2End, self).__init__()
        self.yolo = YOLOv1(num_class, dropout)
        self.retrieval = net.Densenet121(pretrained=True)
        net.embed(self.retrieval, sz_embedding=None)

    def forward(self, images, labels):
        cropped_images, cropped_labels = self.yolo(images, labels)
        output = self.retrieval(cropped_images)
        return output, cropped_labels

    # previous version
    # def forward(self, full_image, cropped_image):
    #     yolo_pred = self.yolo(full_image)
    #     output = []
    #     for crop in cropped_image:
    #         output.append(self.retrieval(crop))
    #     return yolo_pred, output

class YOLOv1(nn.Module):
    def __init__(self, num_class, dropout):
        super(YOLOv1, self).__init__()
        self.dropout = dropout
        self.num_classes = num_class
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
            nn.Linear(4096, 7 * 7 * 5)
        )
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.feature = nn.Sequential(
            self.Conv_7x7_3_64_s2,
            self.Pool,
            self.Conv_3x3_64_192,
            self.Pool,
            self.Conv_1x1_192_128,
            self.Conv_3x3_128_256,
            self.Conv_1x1_256_256,
            self.Conv_3x3_256_512,
            self.Pool,
            self.Conv_1x1_512_256,
            self.Conv_3x3_256_512,
            self.Conv_1x1_512_512,
            self.Conv_3x3_512_1024,
            self.Pool,
            self.Conv_3x3_1024_1024,
            self.Conv_3x3_1024_1024_s2,
            self.Conv_3x3_1024_1024,
            self.Conv_3x3_1024_1024
        )

        self.FC = nn.Sequential(
            self.fc1,
            self.fc2
        )

    def forward(self, x, y):
        out = self.feature(x)
        out = out.reshape(out.size(0), -1)
        out = self.FC(out)
        out = out.reshape((-1, 7, 7, 5))
        out[:,:,:,0] = torch.sigmoid(out[:,:,:,0])
        A = y.cpu().detach().numpy()
        # A = np.delete(A, np.where(~A.any(axis=0))[0], axis=1)

        crop = []
        label = []
        
        for ind, batch in enumerate(out):
            img = torchvision.transforms.ToPILImage(mode='RGB')(x[ind].cpu())

            batch = batch.view(-1, 5).cpu().detach().numpy()
            temp = A[ind][~np.all(A[ind] == 0, axis=-1)]
            ed = euclidean_distances(temp[:,1:-1], batch[:,1:])
            am = np.argmin(ed, axis=1)
            label.append(torch.from_numpy(temp[:, -1]))
            
            for box in batch[am]:
                x_start = max(box[1] * 448 - box[3] * 448 / 2, 0)
                y_start = max(box[2] * 448 - box[4] * 448 / 2, 0)
                x_end = min(x_start + box[3] * 448, 448)
                y_end = min(y_start + box[4] * 448, 448)
                area = (x_start, y_start, x_end, y_end)
                # img = torchvision.transforms.functional.to_pil_image(x)
                cropped_img = img.crop(area)
                cropped_img = cropped_img.resize((224, 224))
                cropped_img = torchvision.transforms.ToTensor()(cropped_img)
                crop.append(cropped_img)


        return torch.stack(crop, 0).cuda(), torch.cat(label, 0).cuda()


def make_embedding_layer(in_features, sz_embedding, weight_init = None):
    embedding_layer = torch.nn.Linear(in_features, sz_embedding)
    if weight_init != None:
        weight_init(embedding_layer.weight)
    return embedding_layer


def bn_inception_weight_init(weight):
    import scipy.stats as stats
    stddev = 0.001
    X = stats.truncnorm(-2, 2, scale=stddev)
    values = torch.Tensor(
        X.rvs(weight.data.numel())
    ).resize_(weight.size())
    weight.data.copy_(values)

def embed(model, sz_embedding, normalize_output = True):
    if sz_embedding is not None:
        model.embedding_layer = make_embedding_layer(
            model.last_linear.in_features,
            sz_embedding,
            weight_init = bn_inception_weight_init
        )

    def forward(x):
        # split up original logits and forward methods
        x = model.features(x)
        x = model.global_pool(x)
        x = x.view(x.size(0), -1)
        if sz_embedding is not None:
            x = model.embedding_layer(x)
        if normalize_output == True:
            x = torch.nn.functional.normalize(x, p = 2, dim = 1)
        return x
    model.forward = forward



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
    # class_label = one_hot(class_pred, target[:,:,:,5:], device)

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

    # obj_class_loss = torch.sum(
    #     objectness_cls_map
    #     *
    #     torch.pow(
    #         class_pred - class_label,
    #         2
    #     )
    # )

    total_loss = obj_coord1_loss + obj_size1_loss + objness1_loss + noobjness1_loss # + obj_class_loss
    total_loss = total_loss / b

    return total_loss, [obj_coord1_loss/b, obj_size1_loss/b, objness1_loss/b, noobjness1_loss/b]#, obj_class_loss/b]
