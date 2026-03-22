# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：trian.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
@Email ：179958974@qq.com
"""
import warnings

import torch

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':

    #model = YOLO(model='2.yaml')

    #model = YOLO(model='DBDown.yaml')
    #model = YOLO(model='DFMS.yaml')
    #model = YOLO(model='3S-RN.yaml')
    #model = YOLO(model='DBDown+DFMS.yaml')
    #model = YOLO(model='DBDown+DFMS+3S-RN.yaml')
    model = YOLO(model='yolov5.yaml')



    #model.load('C:/LJQ/ultralytics-main-ljq/runs/A+B+C+Light+HEAD/train/weights/best.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model.train(data='UAVDT.yaml',
                imgsz=640,
                epochs=200,
                batch=64,
                workers=8,
                #device='1',
                device=[0,1],
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/uavid_result_yolov5',
                single_cls=False,
                cos_lr=True,  # 设置为 True 启用余弦退火学习率调度
                warmup_epochs=10,  # 增加预热期
                amp=False,
                )



