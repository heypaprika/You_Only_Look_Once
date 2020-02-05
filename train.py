# train.py

import os
import warnings

import torch
import torchvision.transforms as transforms
# import visdom
# import wandb

import models
from models import detection_loss_4_yolo

import proxyNS_eu

from torchsummary.torchsummary import summary
from utilities.dataloader import detection_collate, Retrieval_V2_triplet

from utilities.utils import save_checkpoint, create_vis_plot, update_vis_plot, visualize_GT
from utilities.augmentation import Augmenter

from imgaug import augmenters as iaa

warnings.filterwarnings('ignore')

def train(args):
    dataset = args.dataset
    data_path = args.data_path
    class_path = args.class_path
    checkpoint_path = args.checkpoint_path

    input_height = args.input_height
    input_width = args.input_width
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    weight_decay = args.weight_decay
    dropout = args.dropout
    l_coord = args.l_coord
    l_noobj = args.l_noobj
    num_gpus = [i for i in range(args.num_gpus)]
    num_class = args.num_class

    USE_AUGMENTATION = args.use_augmentation
#     USE_VISDOM = args.use_visdom
#     USE_WANDB = args.use_wandb
    USE_SUMMARY = args.use_summary


    if USE_AUGMENTATION:
        seq = iaa.SomeOf(2, [
            iaa.Multiply((1.2, 1.5)),
            iaa.Affine(
                translate_px={"x":3, "y":10},
                scale=(0.9,0.9)
            ),
            iaa.AdditiveGaussianNoise(scale=0.1*255),
            iaa.CoarseDropout(
                0.02, 
                size_percent=0.15, 
                per_channel=0.5
            ),
            iaa.Affine(rotate=45),
            iaa.Sharpen(alpha=0.5)
        ])
    else:
        seq = iaa.Sequential([])
    
    composed = transforms.Compose([Augmenter(seq)])
        
    # DataLoader
    
    train_dataset = Loader(
        root = data_path, 
        transform = composed, 
        class_path = class_path
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        collate_fn = detection_collate
    )
    
    
    # model

    model = models.End2End(num_class, dropout)

    # model = models.YOLOv1(
    #     num_class, dropout
    # )
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=num_gpus).to(device)
    else:
        model = torch.nn.DataParallel(model)
    
    if USE_SUMMARY :
        summary(model, (3, 448, 448))

    trclasses = range(20)
#     criterion = proxyNS_eu.ProxyNS(args.sz_embedding, trclasses, sigma=args.sigma).to(device)


    # Todo 3 : Make loss functioin
    criterion = None
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr = lr,
        weight_decay = weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, 
        gamma = 0.95
    )
    
    total_step = 0
#     total_train_step = num_epochs * total_step
    
    for epoch in range(1, num_epochs + 1):
        if (epoch == 200) or (epoch == 400) or (epoch == 600) or (epoch == 20000) or (epoch == 30000):
            scheduler.step()
        # crop과 crop_target을 사용X
        # dataset 만들 때, image와 pos_ref path, neg_ref path 필요.
        for i, (images, pos_ref, neg_ref) in enumerate(train_loader):

            total_step += 1
            images = images.to(device)
            labels = labels.to(device)
            
            output_vector = model(images)

            # Todo 2: output vector of pos_ref & neg ref
            pos_ref_vector = None
            neg_ref_vector = None


            # loss, losses = detection_loss_4_yolo(yolo_pred, labels, l_coord, l_noobj, device)
            # loss /= 2
            # coord_loss = losses[0] / 2
            # size_loss = losses[1] / 2
            # objness_loss = losses[2] / 2
            # noobjness_loss = losses[3] / 2
            # class_loss = losses[4]

            loss = criterion(output_vector, pos_ref_vector, neg_ref_vector)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if total_step % 100 == 0:
                print(
                    "epoch: [{}/{}], step:{}, lr:{}, total_loss:{:.4f}, \
                    \ncoord:{:.4f}, size:{:.4f}, objness:{:.4f}, noobjness:{:.4f}, cls:{:.4f}".format(
                        epoch, num_epochs, total_step, 
                        ([param['lr'] for param in optimizer.param_groups])[0],
                        loss.item(), coord_loss, size_loss, objness_loss, noobjness_loss, cls_loss
                    ))
            
            if epoch % 1000 == 0:
                save_checkpoint(
                    {
                        "epoch" : epoch,
                        "arch" : "YoloV1",
                        "state_dict" : model.state.dict(),
                        "optimizer" : optimizer.state.dict()
                    }, 
                    False,
                    filename = os.path.join(
                        checkpoint_path,
                        "ckpt_ep{:.05d}_loss{:.04f}_lr{}.pth.tar".format(
                            epoch, loss.item(), ([param['lr'] for param in optimizer.param_group])[0]
                        )
                    )
                )