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
    model = YOLO(model='DBDown+DFMS+3S-RN.yaml')
    #model = YOLO(model='yolo11.yaml')



    #model.load('C:/LJQ/ultralytics-main-ljq/runs/A+B+C+Light+HEAD/train/weights/best.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model.train(data='Tinyperson.yaml',
                imgsz=640,
                epochs=200,
                batch=12,
                workers=8,
                #device='1',
                device=[0,1],
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/Person_result/yolo12m',
                single_cls=False,
                cos_lr=True,  # 设置为 True 启用余弦退火学习率调度
               # warmup_epochs=10,  # 增加预热期
                amp=False,

                lr0=0.008,  # 保持初始学习率
                lrf=0.002,  # 最终学习率降低（原0.005→0.002），后期精细优化
                weight_decay=0.0004,  # 轻微增强正则化（原0.0003→0.0004），避免过拟合
                momentum=0.95,
                warmup_epochs=5,
                )



