import sys
import os

import torch
from torch.utils.data import Dataset
import numpy as np

from PIL import Image

from convertYolo.Format import YOLO as cvtYOLO
from convertYolo.Format import VOC as cvtVOC

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

class VOC(Dataset):
    IMAGE_FOLDER = "JPEGImages"
    LABEL_FOLDER = "Annotations"
    IMG_EXTENSIONS = ".jpg"
    
    def __init__(self, root, train=True, transform=None, target_transform=None, resize=448, class_path='./voc.names'):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.resizing_size = resize
        self.class_path = class_path
        
        with open(self.class_path) as f:
            self.classes = f.read().splitlines()
        if not self._check_exists():
            raise RuntimeError("Dataset not found.")
        self.data = self.cvtData()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        key = list(self.data[index].keys())[0]
        img = Image.open(key).convert('RGB')
        current_shape = img.size
        img = img.resize((self.resizing_size, self.resizing_size))
        target = self.data[index][key]
        
        if self.transform is not None:
            img, aug_target = self.transform([img, target])
            img = torchvision.transforms.ToTensor()(img)
        
        return img, aug_target, current_shape

    def _check_exists(self):
        print("Image Folder:{}".format(
            os.path.join(self.root, self.IMAGE_FOLDER)
        ))
        print("Label Folder:{}".format(
            os.path.join(self.root, self.LABEL_FOLDER)
        ))
        
        is_exist = (
            os.path.exists(
                os.path.join(self.root, self.IMAGE_FOLDER)
            )
        ) and (
            os.path.exists(
                os.path.join(self.root, self.LABEL_FOLDER)
            )
        )
        
        return is_exist
    
    def cvtData(self):
        result = []
        voc = cvtVOC()
        flag, self.dict_data = voc.parse(os.path.join(self.root, self.LABEL_FOLDER))
        yolo = cvtYOLO(os.path.abspath(self.class_path))
        
        try:
            if flag:
                flag, data = yolo.generate(self.dict_data)
                keys = list(data.keys())
                keys = sorted(keys, key=lambda key: int(key.split("_")[-1]))
                
                for key in keys:
                    contents = list(filter(None, data[key].split("\n")))
                    target = []
                    for i in range(len(contents)):
                        tmp = contents[i].split(" ")
                        for j in range(len(tmp)):
                            tmp[j] = float(tmp[j])
                        target.append(tmp)
                    
                    result.append({
                        os.path.join(
                            self.root, 
                            self.IMAGE_FOLDER, 
                            "".join([key, self.IMG_EXTENSIONS])
                        ) : target
                    })
                
                return result
        except Exception as e:
            raise RuntimeError("Error : {}".format(e))

def detection_collate(batch):
    targets = []
    imgs = []
    sizes = []

    for sample in batch:
        imgs.append(sample[0])
        sizes.append(sample[2])

        np_label = np.zeros((7,7,6), dtype=np.float32)

        for object in sample[1]:
            objectness = 1
            classes = object[0]
            x_ratio = object[1]
            y_ratio = object[2]
            w_ratio = object[3]
            h_ratio = object[4]

            scale_factor = 1 / 7
            grid_x_index = int(x_ratio // scale_factor)
            grid_y_index = int(y_ratio // scale_factor)
            x_offset = x_ratio / scale_factor - grid_x_index
            y_offset = y_ratio / scale_factor - grid_y_index

            np_label[grid_x_index][grid_y_index] = np.array([objectness, x_offset, y_offset, w_ratio, h_ratio, classes])

        label = torch.from_numpy(np_label)
        targets.append(label)

    return torch.stack(imgs, 0), torch.stack(targets, 0), sizes