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
                    default=5)
parser.add_argument('-w',
                    type=str,
                    help='path to weight file')
parser.add_argument('-d',
                    type=str,
                    help='path to dataset')
parser.add_argument('-s',
                    type=int,
                    help='random seed',
                    default=3698)
parser.add_argument('-a',
                    action='store_true',
                    help='use attention or not')
options = parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# GPU-related
gpu_list = [int(x) for x in options.gpu.split(',')]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
cudnn.enabled = True

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
num_class = 2
input_size = (321, 321)

# Create network.
model = Res_Deeplab(num_classes=num_class, use_attn=options.a)
model = load_resnet50_param(model, stop_layer='layer4')
model = nn.DataParallel(model, [0])
model.load_state_dict(torch.load(options.w))
model.cuda()

set_seed(options.s)

valset = Dataset_val(data_dir=options.d, fold=options.fold,
                     input_size=input_size,
                     normalize_mean=IMG_MEAN, normalize_std=IMG_STD)
valloader = data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=4,
                            drop_last=False)

iou_list = []
highest_iou = 0
begin_time = time.time()
with torch.no_grad():
    print('----Evaluation----')
    model = model.eval()

    valset.history_mask_list = [None] * 1000
    best_iou = 0
    for eva_iter in range(options.iter_time):
        save_root = f'viz{options.fold}_{options.a}'
        save_dir = f'{save_root}/{eva_iter}'
        os.makedirs(save_dir, exist_ok=True)
        f = open(
            f'{save_root}/score{options.fold}_{eva_iter}.csv', 'w')
        f.write('support,query,class,score\n')

        all_inter, all_union, all_predict = [0] * 5, [0] * 5, [0] * 5
        for i_iter, batch in enumerate(tqdm.tqdm(valloader)):
            # if i_iter != 55:
            #     continue

            query_rgb, query_mask, support_rgb, support_mask, history_mask, sample_class, index, support_name, query_name = batch

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

            plt.subplot(1, 2, 1)
            plt.imshow(convert_image_np(support_rgb[0].cpu()))
            plt.imshow(support_mask[0][0].cpu(), alpha=0.5)

            plt.subplot(1, 2, 2)
            plt.imshow(convert_image_np(query_rgb[0].cpu()))
            plt.imshow(pred_label[0].cpu(), alpha=0.5)

            plt.tight_layout()
            plt.savefig(f'{save_dir}/{i_iter:03d}')
            # plt.show()
            plt.close()

            _, pred_label = torch.max(pred, 1)
            inter_list, union_list, _, num_predict_list = get_iou_v1(
                query_mask, pred_label)
            f.write(
                f'{support_name[0]},{query_name[0]},{sample_class[0]},{float(inter_list[0])/union_list[0]}\n')
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

        f.close()
    print('IOU for this epoch: %.4f' % (best_iou))

epoch_time = time.time() - begin_time
print('This epoch takes:', epoch_time, 'second')
