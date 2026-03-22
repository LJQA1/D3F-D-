
import warnings
import torch

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载自定义模型配置
    model = YOLO(model='DBDown+DFMS+3S-RN.yaml')

    # 从零训练（不加载预训练权重）
    model.train(
        data='VisDrone.yaml',

        #data='AI-TOD.yaml',

        imgsz=640,
        epochs=300,
        batch=12,
        workers=8,
        # device=[0, 1],
        optimizer='SGD',
        close_mosaic=20,
        resume=False,
        project='runs/1',
        single_cls=False,
        cos_lr=True,
        lr0=0.008,
        lrf=0.005,
        warmup_epochs=5,
        warmup_momentum=0.85,
        warmup_bias_lr=0.05,
        momentum=0.95,
        weight_decay=0.0003,
        box=8.0,
        cls=0.8,
        dfl=1.8,
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.08,
        scale=0.4,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.05,
        patience=50,
        val=True,
        save=True,
        save_period=10,
        amp=False,
        multi_scale=False,
    )