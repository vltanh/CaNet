from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os.path as osp
from utils import *
import time
import torch.nn.functional as F
import tqdm
import random
import argparse
from dataset_mask_train import Dataset as Dataset_train
from new_dataset_mask_val import Dataset as Dataset_val
import os
import torch
from one_shot_network import Res_Deeplab
import torch.nn as nn
import numpy as np


parser = argparse.ArgumentParser()


parser.add_argument('-lr',
                    type=float,
                    help='learning rate',
                    default=0.00025)

parser.add_argument('-prob',
                    type=float,
                    help='dropout rate of history mask',
                    default=0.7)


parser.add_argument('-bs',
                    type=int,
                    help='batchsize',
                    default=4)

parser.add_argument('-bs_val',
                    type=int,
                    help='batchsize for val',
                    default=1)


parser.add_argument('-fold',
                    type=int,
                    help='fold',
                    default=0)


parser.add_argument('-gpu',
                    type=str,
                    help='gpu id to use',
                    default='0,1')


parser.add_argument('-iter_time',
                    type=int,
                    default=1)

parser.add_argument('-w',
                    type=str)

parser.add_argument('-d',
                    type=str)

parser.add_argument('-s',
                    type=int,
                    default=3698)

parser.add_argument('-a', action='store_true')

options = parser.parse_args()

SEED = options.s
data_dir = options.d


# set gpus
gpu_list = [int(x) for x in options.gpu.split(',')]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def set_determinism():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(SEED)
# set_determinism()

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
num_class = 2
num_epoch = 1
learning_rate = options.lr  # 0.000025#0.00025
input_size = (321, 321)
batch_size = options.bs
weight_decay = 0.0005
momentum = 0.9
power = 0.9

cudnn.enabled = True

# Create network.
model = Res_Deeplab(num_classes=num_class)
model = load_resnet50_param(model, stop_layer='layer4')
model = nn.DataParallel(model, [0])
model.load_state_dict(torch.load(options.w))

valset = Dataset_val(data_dir=data_dir, fold=options.fold, input_size=input_size, normalize_mean=IMG_MEAN,
                     normalize_std=IMG_STD)
valloader = data.DataLoader(valset, batch_size=options.bs_val, shuffle=False, num_workers=4,
                            drop_last=False)

iou_list = []  # track validaiton iou
highest_iou = 0

model.cuda()
begin_time = time.time()
with torch.no_grad():
    print('----Evaluation----')
    model = model.eval()

    valset.history_mask_list = [None] * 1000
    best_iou = 0
    for eva_iter in range(options.iter_time):
        all_inter, all_union, all_predict = [0] * 5, [0] * 5, [0] * 5
        for i_iter, batch in enumerate(tqdm.tqdm(valloader)):
            if i_iter != 55:
                continue

            query_rgb, query_mask, support_rgb, support_mask, history_mask, sample_class, index = batch

            query_rgb = (query_rgb).cuda(0)
            support_rgb = (support_rgb).cuda(0)
            support_mask = (support_mask).cuda(0)
            # change formation for crossentropy use
            query_mask = (query_mask).cuda(0).long()

            # remove the second dim,change formation for crossentropy use
            query_mask = query_mask[:, 0, :, :]
            history_mask = (history_mask).cuda(0)

            pred = model(query_rgb, support_rgb,
                         support_mask, history_mask)
            pred_softmax = F.softmax(pred, dim=1).data.cpu()

            # update history mask
            for j in range(support_mask.shape[0]):
                sub_index = index[j]
                valset.history_mask_list[sub_index] = pred_softmax[j]

                pred = nn.functional.interpolate(pred, size=query_mask.shape[-2:], mode='bilinear',
                                                 align_corners=True)  # upsample  # upsample

            _, pred_label = torch.max(pred, 1)

            # plt.subplot(1, 2, 1)
            # plt.imshow(convert_image_np(support_rgb[0].cpu()))
            # plt.imshow(support_mask[0][0].cpu(), alpha=0.5)

            # plt.subplot(1, 2, 2)
            # plt.imshow(convert_image_np(query_rgb[0].cpu()))
            # plt.imshow(pred_label[0].cpu(), alpha=0.5)

            # plt.tight_layout()
            # os.makedirs(
            #     f'viz{options.fold}/{eva_iter}', exist_ok=True)
            # plt.savefig(f'viz{options.fold}/{eva_iter}/{i_iter:03d}')
            # # plt.show()
            # plt.close()

            _, pred_label = torch.max(pred, 1)
            inter_list, union_list, _, num_predict_list = get_iou_v1(
                query_mask, pred_label)
            for j in range(query_mask.shape[0]):  # batch size
                all_inter[sample_class[j] -
                          (options.fold * 5 + 1)] += inter_list[j]
                all_union[sample_class[j] -
                          (options.fold * 5 + 1)] += union_list[j]

        IOU = [0] * 5

        for j in range(5):
            IOU[j] = all_inter[j] / all_union[j]

        mean_iou = np.mean(IOU)
        print(IOU)
        print('IOU:%.4f' % (mean_iou))
        if mean_iou > best_iou:
            best_iou = mean_iou
    print('IOU for this epoch: %.4f' % (best_iou))

epoch_time = time.time() - begin_time
print('This epoch takes:', epoch_time, 'second')
