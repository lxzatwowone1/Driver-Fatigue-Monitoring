#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by Liuxiaozhe on 2020/9/25
import warnings
warnings.filterwarnings("ignore")

import torch
from torchvision import transforms
import numpy as np
import cv2
from pfld import PFLDInference, MTCNN
from utils import getDistance, glotPosecube


class FaceAttribute:
    def __init__(self):
        #人脸检测
        self.mtcnn = MTCNN()
        #人脸关键点定位模型
        self.pfldmodel = PFLDInference().cuda()
        self.pfldmodel.load_state_dict(torch.load("weights/keypoints.pth"))
        self.pfldmodel = self.pfldmodel.cuda()
        self.pfldmodel.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def main(self,img):
        transform = transforms.Compose([transforms.ToTensor()])
        with torch.no_grad():
            height, width = img.shape[:2]
            img_det = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            det = self.mtcnn.detect_face(img_det)
            # '''
            # [{'box': (160, 48, 322, 270), 'cls': array([0.9666081], dtype=float32),
            #   'pts': {'leye': (214, 136), 'reye': (276, 121), 'nose': (257, 159), 'lmouse': (238, 208),
            #           'rmouse': (288, 197)}}]
            # '''
            for i in range(len(det)): #单张图片人脸数量
                box = det[i]['box']   #人脸框tuple
                #cls = result[i]['cls']  #置信度ndarry
                pts = det[i]['pts']   #五官坐标dict
                x1, y1, x2, y2 = box     #左上右下
                dis = y2 - y1
                #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 25)) #天蓝色人脸框
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                size_w = int(max([w, h])*0.9)
                size_h = int(max([w, h]) * 0.9)
                cx = x1 + w//2
                cy = y1 + h//2
                x1 = cx - size_w//2
                x2 = x1 + size_w
                y1 = cy - int(size_h * 0.4)
                y2 = y1 + size_h
                left = 0
                top = 0
                bottom = 0
                right = 0
                if x1 < 0:
                    left = -x1
                if y1 < 0:
                    top = -y1
                if x2 >= width:
                    right = x2 - width
                if y2 >= height:
                    bottom = y2 - height

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                cropped = img[y1:y2, x1:x2]  #裁剪出的人脸
                # print(cropped.shape)
                # np_img = img[int(y1/1):y2+int(y1/1),int(x1/1):x2+int(x1/1)]
                # cv2.imshow(str(numa),np_img)
                cropped = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
                input = cv2.resize(cropped, (112, 112))
                input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
                input = transform(input).unsqueeze(0).cuda()
                pose, landmarks = self.pfldmodel(input)
                #poses = pose.cpu().detach().numpy()[0] * 180 / np.pi
                # 长度3 pitch是围绕X轴旋转,也叫做俯仰角。 yaw是围绕Y轴旋转,也叫偏航角。 roll是围绕Z轴旋转,也叫翻滚角
                pre_landmark = landmarks[0]
                pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [size_w, size_h]  # 长度98
                # cv2.rectangle(img,(x1, y1), (x2, y2),(255,0,0)) #蓝色正方形
                fatigue = []
                for num, (x, y) in enumerate(pre_landmark.astype(np.int32)):
                    #cv2.circle(img, (x1 - left + x, y1 - bottom + y), 1, (255, 255, 0), 1)
                    #if 59 < num < 76 or num in [96,97]: #眼眶坐标
                    #眼镜轮廓坐标
                    #             62                     70
                    #       60         64          68          72
                    #             66                     74

                    if num in [60, 62, 64, 66, 68, 70, 72, 74]:
                        cv2.circle(img, (x1 - left + x, y1 - bottom + y), 1, (255, 255, 0), 1)
                        fatigue.append((x, y))
                # print(fatigue)
                rightrow = getDistance(fatigue[0], fatigue[2])
                rightcol = getDistance(fatigue[1],fatigue[3])
                leftrow = getDistance(fatigue[4],fatigue[6])
                leftcol = getDistance(fatigue[5],fatigue[7])
                numerator = rightcol+leftcol
                denominator = rightrow+leftrow
                distance = numerator/denominator
                print('dis:'+str(distance))
                text ,color= 'eyes closed!' if distance < 0.17 else 'eyes opened',\
                             (0, 0, 255) if distance < 0.17 else (0, 255, 0)



                img = cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                # another way 计算
                # eye = rightcol + leftcol
                # if eye / dis < 0.03:
                # print('dis:' + str(distance))
                # print('eyes closed!')
                # else:
                #     print('ok')

                # plotPosecube(img, poses[0], poses[1], poses[2], tdx=pts['nose'][0], tdy=pts['nose'][1],
                #        size=(x2 - x1) // 2)
            cv2.imshow('example', img)
            cv2.waitKey(0)






if __name__ == "__main__":
    import glob
    F = FaceAttribute()
    f = glob.glob('D:\points/face\FaceAttributeClassiry-master/test/*')
    for i in f:
        img = cv2.imread(i)
        F.main(img)




