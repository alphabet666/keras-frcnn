# -*- coding: utf-8 -*-
# https://github.com/yhenon/keras-frcnn
import cv2
import os
from skimage import measure,color
import random

text = open('trainingData.txt','w')
imgDir = "D:\\dengh\\mitochondria public\\training_groundtruth\\"
imgList = os.listdir(imgDir)
imgNum = len(imgList)
for i in range (imgNum):
    imgName = imgList[i]
    img = cv2.imread(imgDir+imgName, cv2.IMREAD_COLOR)
    labels = measure.label(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), connectivity=2)
    rect = measure.regionprops(labels)
    for i_rect in range(len(rect)):
        [r_min, c_min, r_max, c_max] = rect[i_rect].bbox
        # # 画外接框
        # cv2.rectangle(img, (c_min,r_min),(c_max,r_max),(255,0,0),3)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # 适当扩大边界框，并存入txt
        r_min = (max(1, r_min - random.randrange(5, 20)))
        c_min = (max(1, c_min - random.randrange(5, 20)))
        r_max = (min(img.shape[0], r_max + random.randrange(5, 10)))
        c_max = (min(img.shape[1], c_max + random.randrange(5, 10)))
        # text.write(str(r_min)+' '+str(c_min)+' '+str(r_max)+' '+str(c_max)+ '\n')
        print(imgDir[:-13] + '\\' + imgName[0:8] + imgName[-7:] + ',' + str(c_min) + ',' + str(r_min) + ',' + str(c_max) + ',' + str(r_max) + ',mitochondria', file = text)



