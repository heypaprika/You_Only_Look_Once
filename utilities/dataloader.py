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

        crop = []

        if aug_target is not None:
            for box in aug_target:
                x_start = max(box[1] * 448 - box[3] * 448 / 2, 0)
                y_start = max(box[2] * 448 - box[4] * 448 / 2, 0)
                x_end = min(x_start + box[3] * 448, 448)
                y_end = min(y_start + box[4] * 448, 448)
                area = (x_start, y_start, x_end, y_end)
                cropped_img = img.crop(area)
                cropped_img = cropped_img.resize((224, 224))
                cropped_img = torchvision.transforms.ToTensor()(cropped_img)
                crop.append(cropped_img)

        if len(crop) is 0:
            crop = None
        else:
            crop = torch.stack(crop)

        if self.transform is not None:
            img = torchvision.transforms.ToTensor()(img)

        return img, aug_target, current_shape, crop

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

    crop = []
    crop_target = []
    for sample in batch:
        crop.append(sample[3])
        imgs.append(sample[0])
        sizes.append(sample[2])

        # bounding box : 3
        np_label = np.zeros((49, 6), dtype=np.float32)
        array_label = []
        for idx, object in enumerate(sample[1]):
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

            # array_label.append([objectness, x_offset, y_offset, w_ratio, h_ratio, classes])

            np_label[idx] = np.array([objectness, x_offset, y_offset, w_ratio, h_ratio, classes])
            crop_target.append(torch.Tensor([classes]))

        label = torch.Tensor(np_label)
        targets.append(label)

    return torch.stack(imgs, 0), torch.stack(targets, 0), sizes, torch.cat(crop, 0), torch.stack(crop_target, 0)