import torchvision.transforms as tvtf
from PIL import Image
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from one_shot_network import Res_Deeplab
from utils import load_resnet50_param, convert_image_np
import random
# plt.rcParams["figure.figsize"] = (15, 5)


parser = argparse.ArgumentParser()
parser.add_argument('--gpus', default='0')
parser.add_argument('--weight')
parser.add_argument('--refid')
parser.add_argument('--queid')
parser.add_argument('--classid', type=int)
parser.add_argument('--niters', default=5, type=int)
parser.add_argument('--a', action='store_true')
args = parser.parse_args()

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


model = Res_Deeplab(num_classes=2, use_attn=args.a)
model = load_resnet50_param(model, stop_layer='layer4')
model = nn.DataParallel(model, [0])
model.load_state_dict(torch.load(args.weight))
model.cuda()
model.eval()

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
root = '../data/PASCAL-5i/'
ref_img_path = root + '/JPEGImages/' + args.refid + '.jpg'
ref_mask_path = root + '/Annotations/' + \
    CLASSES[args.classid - 1] + '/' + args.refid + '.png'
que_img_path = root + '/JPEGImages/' + args.queid + '.jpg'

niters = args.niters

with torch.no_grad():
    ref_img = Image.open(ref_img_path).convert('RGB')
    ref_mask = Image.open(ref_mask_path).convert('P')
    query_img = Image.open(que_img_path).convert('RGB')

    tf = tvtf.Compose([
        tvtf.ToTensor(),
        tvtf.Normalize(IMG_MEAN, IMG_STD),
    ])
    ref_img = tf(ref_img).unsqueeze(0).cuda()
    ref_mask = torch.FloatTensor(
        np.array(ref_mask) > 0).unsqueeze(0).unsqueeze(0).cuda()
    query_img = tf(query_img).unsqueeze(0).cuda()
    history_mask = torch.zeros(1, 2, 41, 41).cuda()

    fig, ax = plt.subplots(1, niters+1)

    ax[0].imshow(convert_image_np(ref_img[0].cpu()))
    ax[0].imshow(ref_mask[0, 0].cpu(), alpha=0.5)
    # ax[0].set_title('Reference')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    for i in range(niters):
        out = model(query_img, ref_img, ref_mask, history_mask)
        history_mask = torch.softmax(out, dim=1)
        pred = F.interpolate(history_mask, size=query_img.shape[-2:],
                             mode='bilinear',
                             align_corners=True)
        pred = torch.argmax(pred, dim=1)

        ax[1+i].imshow(convert_image_np(query_img[0].cpu()))
        ax[1+i].imshow(pred[0].cpu(), alpha=0.5)
        # ax[1+i].set_title(f'Query')
        ax[1+i].set_xticks([])
        ax[1+i].set_yticks([])

    fig.tight_layout()
    plt.show()
    plt.close()
