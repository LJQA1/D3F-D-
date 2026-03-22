from ultralytics import YOLO

if "__main__" == __name__:
    model = YOLO("C:/LJQ/ultralytics-main-ljq/runs/AA+BB+CC+PRNet/train5/weights/best.pt")
    #model = YOLO("C:/LJQ/ultralytics-main-ljq/runs/DBDown+DFMS/train2/weights/best.pt")
    #model = YOLO("C:/LJQ/ultralytics-main-ljq/runs/aitod_new/train3/weights/best.pt")   #ai-tod

    #model = YOLO("C:/LJQ/ultralytics-main-ljq/runs/DBDown+DFMS+3S-RN_op_map50-95(AI-TOD_tiaocan)/train/weights/best.pt")  # ai-tod

   # print(model)

    results = model.val(
        data='VisDrone.yaml',
        #data='AI-TOD.yaml',
        batch=12,
        imgsz=640,
        device="0,1",
        workers=8,
        name='VisDrone_12change_neck',
        #conf=0.35,  # (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
        conf=0.35,
        iou=0.45  # (float) intersection over union (IoU) threshold for NMS
    )





