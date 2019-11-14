# train test를 실행하는 함수. argparse를 이용하여 hyperparameter 등과 같은 argument에 대한 값을 받아서 train 혹은 test로 값을 넘겨주는 역할을 한다.

import argparse

from train import train
from test import test

parser = argparse.ArgumentParser(description='Yolo V1')
parser.add_argument('--mode', type=str, help='train test', default='train')

# Data
parser.add_argument('--dataset', type=str, default='voc')
parser.add_argument('--data_path', type=str, default='/home/yangho/dev/dataset/VOCdevkit/VOC2007')
parser.add_argument('--class_path', type=str, default='./names/voc.names')
parser.add_argument('--checkpoint_path', type=str, default='./')

# Input
parser.add_argument('--input_height', type=int, default=448)
parser.add_argument('--input_width', type=int, default=448)

# Train / Test
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=16000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--l_coord', type=int, default=5)
parser.add_argument('--l_noobj', type=float, default=0.5)
parser.add_argument('--num_gpus', type=int, default=1)

# flag
parser.add_argument('--use_augmentation', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--use_visdom', type=lambda x:(str(x).lower() == 'true'), default=False)
parser.add_argument('--use_wandb', type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--use_summary', type=lambda x: (str(x).lower() == 'true'), default=False)

# develop
parser.add_argument('--num_class', type=int, default=20)

args = parser.parse_args()

def main():
    
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)

if __name__ == "__main__":
    main()