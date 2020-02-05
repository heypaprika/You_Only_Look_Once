import sys
import os
from itertools import combinations
from random import choice

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

class Retrieval_V2_triplet(Dataset):
    def __init__(self, data_path, transform, desired_size=250):
        self.label_list = []
        self.imgpath_list = []
        self.label_idx = 0

        for root, dirs, files in os.walk(data_path):
            if not files:
                continue
            for filename in files:
                img_path = os.path.join(root, filename)
                if img_path.split('.')[-1] in set(['jpg', 'JPG', 'png']):
                    self.label_list.append(self.label_idx)
                    self.imgpath_list.append(img_path)
                else:
                    print('image fault {}'.format(img_path))
                    continue
            self.label_idx += 1
    
        # Triplet
        # ToDo 1: make Triplet lists

        self.q_list = []
        self.pos_ref_list = []
        self.neg_ref_list = []

        data_list = [[] for _ in range(1000)]
        data_set = set()


        for path in self.imgpath_list:
            data_list[int(path.split("/")[-2])].append(path)
            data_set.add(path.split("/")[-2])

        classnum_list = list(data_set)
        comb_list = []
        third_path = []

        for classnum, data in enumerate(data_list):
            if len(data) == 0: continue
            comb = list(combinations(data, 2))
            comb_list += comb
            for _ in range(len(comb)):
                other_class = choice([int(i) for i in classnum_list if int(i) not in [classnum]])
                if int(other_class) == classnum:print("error"); return
                third_path.append(choice(data_list[other_class]))

        self.q_list, self.pos_ref_list = list(map(list, zip(*comb_list)))
        self.neg_ref_list = third_path

    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, idx):

        # ToDo 1-1: path to img

        im = Image.open(self.imgpath_list[idx])
        im = self.transform(im)
        return im
        # , pos_ref, neg_ref
    
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