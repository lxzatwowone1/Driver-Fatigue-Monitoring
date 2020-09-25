#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from hdface.hdface import hdface_detector


def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inp * expand_ratio,
                inp * expand_ratio,
                3,
                stride,
                1,
                groups=inp * expand_ratio,
                bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PFLDInference(nn.Module):
    def __init__(self):
        super(PFLDInference, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv3_1 = InvertedResidual(64, 64, 2, False, 2)

        self.block3_2 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_3 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_4 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_5 = InvertedResidual(64, 64, 1, True, 2)

        self.conv4_1 = InvertedResidual(64, 128, 2, False, 2)

        self.conv5_1 = InvertedResidual(128, 128, 1, False, 4)
        self.block5_2 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_3 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_4 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_5 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_6 = InvertedResidual(128, 128, 1, True, 4)

        self.conv6_1 = InvertedResidual(128, 16, 1, False, 2)  # [16, 14, 14]

        self.conv7 = conv_bn(16, 32, 3, 2)  # [32, 7, 7]
        self.conv8 = nn.Conv2d(32, 128, 7, 1, 0)  # [128, 1, 1]
        self.bn8 = nn.BatchNorm2d(128)

        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(176, 196)
        self.fc_aux = nn.Linear(176, 3)

        self.conv1_aux = conv_bn(64, 128, 3, 2)
        self.conv2_aux = conv_bn(128, 128, 3, 1)
        self.conv3_aux = conv_bn(128, 32, 3, 2)
        self.conv4_aux = conv_bn(32, 128, 7, 1)
        self.max_pool1_aux = nn.MaxPool2d(3)
        self.fc1_aux = nn.Linear(128, 32)
        self.fc2_aux = nn.Linear(32 + 176, 3)

    def forward(self, x):  # x: 3, 112, 112
        x = self.relu(self.bn1(self.conv1(x)))  # [64, 56, 56]
        x = self.relu(self.bn2(self.conv2(x)))  # [64, 56, 56]
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)

        x = self.conv4_1(out1)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)
        x = self.conv6_1(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.relu(self.conv8(x))
        x3 = x3.view(x1.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)


        aux = self.conv1_aux(out1)
        aux = self.conv2_aux(aux)
        aux = self.conv3_aux(aux)
        aux = self.conv4_aux(aux)
        aux = self.max_pool1_aux(aux)
        aux = aux.view(aux.size(0), -1)
        aux = self.fc1_aux(aux)
        aux = torch.cat([aux, multi_scale], 1)
        pose = self.fc2_aux(aux)

        return pose, landmarks

class MTCNN(hdface_detector):
    def __init__(self,
                 min_face_size=12,
                 use_cuda=True,
                 threshold=[0.5, 0.5, 0.8],
                 scale_factor=0.7,
                 stage = 3,
                 stride=2,
                 ):
        super().__init__(min_face_size,use_cuda,threshold,scale_factor,stage,stride)

    def detect_face(self,img):
        """Detect face over image
        """
        result = []
        boxes_align = None
        landmark_align =None

        # pnet
        if self.pnet_detector:
            boxes, boxes_align = self.detect_pnet(img)
            if boxes_align is None:
                return []
        else:
            return []
        # rnet
        if self.rnet_detector:
            boxes, boxes_align = self.detect_rnet(img, boxes_align)
            if boxes_align is None:
                return []

        else:
            return boxes_align
        # onet
        if self.onet_detector:
            boxes_align, landmark_align = self.detect_onet(img, boxes_align)
            if boxes_align is None:
                return []

            if len(boxes_align) > 0:
                lds_img = img.copy()
                bboxs = boxes_align
                sq_bboxes = self.square_bbox(bboxs).astype(np.int32)
                for i in range(bboxs.shape[0]):
                    single = {}
                    sq_bboxes[i][1] = max(0, sq_bboxes[i][1])
                    sq_bboxes[i][0] = max(0, sq_bboxes[i][0])
                    sq_bboxes[i][3] = min(lds_img.shape[0] - 1, sq_bboxes[i][3])
                    sq_bboxes[i][2] = min(lds_img.shape[1] - 1, sq_bboxes[i][2])
                    patch = lds_img[sq_bboxes[i][1]:sq_bboxes[i][3], sq_bboxes[i][0]:sq_bboxes[i][2], :]
                    cls, reg, landmarks = self.detect_lnet(patch)
                    # box_align
                    xmin = int(reg[0] * patch.shape[0] + sq_bboxes[i][0])
                    ymin = int(reg[1] * patch.shape[1] + sq_bboxes[i][1])
                    xmax = int(reg[2] * patch.shape[0] + sq_bboxes[i][2])
                    ymax = int(reg[3] * patch.shape[1] + sq_bboxes[i][3])

                    lds = []
                    for j in range(5):
                        x = int(landmarks[2 * j] * patch.shape[1] + sq_bboxes[i][0])
                        y = int(landmarks[2 * j + 1] * patch.shape[0] + sq_bboxes[i][1])
                        lds.append((x, y))

                    single = {
                        'box': (xmin, ymin, xmax, ymax),
                        'cls': cls,
                        'pts': {
                            'leye': lds[0],
                            'reye': lds[1],
                            'nose': lds[2],
                            'lmouse': lds[3],
                            'rmouse': lds[4],
                        }
                    }
                    result.append(single)
            #print("time cost " + '{:.3f}'.format(t1+t2+t3) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2, t3))
            return result
        else:
            return boxes_align


if __name__ == '__main__':
    input = torch.randn(1, 3, 112, 112)
    plfd_backbone = PFLDInference()
    angle, landmarks = plfd_backbone(input)
    print(plfd_backbone)
    print("angle.shape:{0:}, landmarks.shape: {1:}".format(
        angle.shape, landmarks.shape))
