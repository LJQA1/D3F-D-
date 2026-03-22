import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = RTDETR('runs/Visdrone_train/R18-SAFM_down1280_V32/weights/best.pt') # select your model.pt path
    #model = YOLO("C:/LJQ/ultralytics-main-ljq/runs/DBDown+DFMS+3S-RN_op_map50-95(AI-TOD_tiaocan)/train3/weights/best.pt")
    model = YOLO("C:/LJQ/ultralytics-main-ljq/runs/DBDown+DFMS+3S-RN_op_map50-95(AI-TOD_tiaocan)/train/weights/best.pt")
    model.predict(source='C:/Users/admin/Desktop/DSQ-1',
                  conf=0.60,
                  imgsz=640,
                  project='C:/Users/admin/Desktop/2',
                  name='VIS-v1-0.70',
                  save=True,
                #   visualize=True # visualize model features maps
                  )