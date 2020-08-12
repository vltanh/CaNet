import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import math
from utils import convert_image_np

# code of dilated convolution part is referenced from https://github.com/speedinghzl/Pytorch-Deeplab

affine_par = True


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,
                               stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()

    def forward(self, m_k, m_v, q_k):
        # m_k: B, Dk, Hm, Wm
        # m_v: B, Dv, Hm, Wm
        # q_k: B, Dk, Hq, Wq

        B, Dk, Hm, Wm = m_k.size()
        _,  _, Hq, Wq = q_k.size()
        _, Dv,  _,  _ = m_v.size()

        mk = m_k.reshape(B, Dk, Hm*Wm)  # mk: B, Dk, Hm*Wm
        mk = torch.transpose(mk, 1, 2)  # mk: B, Hm*Wm, Dk

        qk = q_k.reshape(B, Dk, Hq*Wq)  # qk: B, Dk, Hq*Wq

        p = torch.bmm(mk, qk)  # p: B, Hm*Wm, Hq*Wq
        p = p / math.sqrt(Dk)  # p: B, Hm*Wm, Hq*Wq
        p = F.softmax(p, dim=1)  # p: B, Hm*Wm, Hq*Wq

        mv = m_v.reshape(B, Dv, Hm*Wm)  # mv: B, Dv, Hm*Wm
        mem = torch.bmm(mv, p)  # B, Dv, Hq*Wq
        mem = mem.reshape(B, Dv, Hq, Wq)  # B, Dv, Hq, Wq

        return mem, p


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, use_attn):
        self.inplanes = 64
        self.use_attn = use_attn
        super(ResNet, self).__init__()

        # ResNet-50 (Deeplab variant)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block,  64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1],
                                       stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2],
                                       stride=1, dilation=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        # Key-Value generator
        if not self.use_attn:
            self.layer5 = nn.Sequential(
                nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=3,
                          stride=1, padding=2, dilation=2, bias=True),
                nn.ReLU(),
                nn.Dropout2d(p=0.5),
            )
        else:
            self.layer5_K = nn.Sequential(
                nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=3,
                          stride=1, padding=2, dilation=2, bias=True),
                nn.ReLU(),
                nn.Dropout2d(p=0.5),
            )

            self.layer5_V = nn.Sequential(
                nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=3,
                          stride=1, padding=2, dilation=2, bias=True),
                nn.ReLU(),
                nn.Dropout2d(p=0.5),
            )

        # Memory augmented feature map post-process
        self.layer55 = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2, out_channels=256, kernel_size=3,
                      stride=1, padding=2, dilation=2, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )

        # ASPP
        self.layer6_0 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )

        self.layer6_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )

        self.layer6_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3,
                      stride=1, padding=6, dilation=6, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )

        self.layer6_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3,
                      stride=1, padding=12, dilation=12, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )

        self.layer6_4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3,
                      stride=1, padding=18, dilation=18, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )

        # Decoder (Iterative Optimization Module)
        self.residule1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256+2, 256, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3,
                      stride=1, padding=1, bias=True)
        )

        self.residule2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3,
                      stride=1, padding=1, bias=True)
        )

        self.residule3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3,
                      stride=1, padding=1, bias=True)
        )

        # Prediction
        self.layer9 = nn.Conv2d(
            256, num_classes, kernel_size=1, stride=1, bias=True)

        # Memory
        self.memory = Memory()

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, downsample=None):
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par)
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes,
                            stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, query_rgb, support_rgb, support_mask, history_mask, vis_attn=False):
        ref_img = support_rgb.clone()
        ref_mask = support_mask.clone()
        query_img = query_rgb.clone()
        #print('Input:', query_img.shape)

        # === Query feature extraction
        query_rgb = self.conv1(query_rgb)
        #print('Conv 0:', query_rgb.shape)
        query_rgb = self.bn1(query_rgb)
        query_rgb = self.relu(query_rgb)
        query_rgb = self.maxpool(query_rgb)
        #print('Layer 0:', query_rgb.shape)
        query_rgb = self.layer1(query_rgb)
        #print('Layer 1:', query_rgb.shape)
        query_rgb = self.layer2(query_rgb)
        #print('Layer 2:', query_rgb.shape)
        query_feat_layer2 = query_rgb
        query_rgb = self.layer3(query_rgb)
        #print('Layer 3:', query_rgb.shape)
        # query_rgb = self.layer4(query_rgb)
        query_rgb_ = torch.cat([query_feat_layer2, query_rgb], dim=1)
        feature_size = query_rgb_.shape[-2:]
        #print('Encoder:', query_rgb_.shape)

        # === Query key-value generation
        if not self.use_attn:
            query_rgb = self.layer5(query_rgb_)
        else:
            query_rgb_K = self.layer5_K(query_rgb_)
            query_rgb_V = self.layer5_V(query_rgb_)
        #print('Key/Value:', query_rgb_K.shape)

        # === Reference feature extraction
        support_rgb = self.conv1(support_rgb)
        support_rgb = self.bn1(support_rgb)
        support_rgb = self.relu(support_rgb)
        support_rgb = self.maxpool(support_rgb)
        support_rgb = self.layer1(support_rgb)
        support_rgb = self.layer2(support_rgb)
        support_feat_layer2 = support_rgb
        support_rgb = self.layer3(support_rgb)
        #support_rgb = self.layer4(support_rgb)
        support_rgb_ = torch.cat([support_feat_layer2, support_rgb], dim=1)

        # === Reference key-value generation
        if not self.use_attn:
            support_rgb = self.layer5(support_rgb_)
        else:
            support_rgb_K = self.layer5_K(support_rgb_)
            support_rgb_V = self.layer5_V(support_rgb_)

        # === Dense comparison OR Memory read
        support_mask = F.interpolate(support_mask, support_rgb.shape[-2:],
                                     mode='bilinear', align_corners=True)
        if not self.use_attn:
            z = support_mask * support_rgb
            z, viz = self.memory(z, z, query_rgb)
            out = torch.cat([query_rgb, z], dim=1)
        else:
            z_K = support_mask * support_rgb_K
            z_V = support_mask * support_rgb_V
            z, viz = self.memory(z_K, z_V, query_rgb_K)
            out = torch.cat([query_rgb_V, z], dim=1)
        #print(out.shape)

        #import matplotlib.pyplot as plt
        #for i in range(viz.size(2)):
        #    m = torch.zeros(query_rgb.shape[-2], query_rgb.shape[-1])
        #    m[i // query_rgb.shape[-1], i % query_rgb.shape[-1]] = 1
        #    m = F.interpolate(m.unsqueeze(0).unsqueeze(
        #        0), (query_img.shape[-2], query_img.shape[-1])).squeeze(0).squeeze(0)
        #    # f = query_img[0].permute(1, 2, 0).detach().cpu()
        #    plt.figure(figsize=(16, 8), dpi=100)
        #    plt.subplot(1, 2, 1)
        #    plt.imshow(convert_image_np(query_img[0].cpu()))
        #    plt.imshow(m, alpha=0.5)
        #    plt.xticks([])
        #    plt.yticks([])
        #    plt.subplot(1, 2, 2)
        #    v = viz[0, :, i].reshape(
        #        support_rgb.shape[-2], support_rgb.shape[-1]).detach().cpu()
        #    v = F.interpolate(v.unsqueeze(
        #        0).unsqueeze(0), (ref_img.shape[-2], ref_img.shape[-1])).squeeze(0).squeeze(0)
        #    f = ref_img[0].detach().cpu()
        #    plt.imshow(convert_image_np(f))
        #    plt.imshow(v, alpha=0.5)
        #    plt.xticks([])
        #    plt.yticks([])
        #    plt.tight_layout()
        #    plt.savefig(f'viz/{i:04d}')
        #    # plt.show()
        #    plt.close()

        # === Decoder
        # Residue blocks
        history_mask = F.interpolate(history_mask, feature_size,
                                     mode='bilinear', align_corners=True)
        out = self.layer55(out)
        out_plus_history = torch.cat([out, history_mask], dim=1)
        out = out + self.residule1(out_plus_history)
        out = out + self.residule2(out)
        out = out + self.residule3(out)
        #print('ResBlocks:', out.shape)

        # ASPP
        global_feature = F.avg_pool2d(out, kernel_size=feature_size)
        global_feature = self.layer6_0(global_feature)
        global_feature = global_feature.expand(-1, -1,
                                               feature_size[0], feature_size[1])
        out = torch.cat([global_feature,
                         self.layer6_1(out),
                         self.layer6_2(out),
                         self.layer6_3(out),
                         self.layer6_4(out)],
                        dim=1)
        out = self.layer7(out)
        #print('ASPP:', out.shape)

        # === Prediction
        out = self.layer9(out)
        #print('Output:', out.shape)

        if vis_attn:
            return out, viz
        else:
            return out


def Res_Deeplab(num_classes=2, use_attn=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, use_attn)
    return model
