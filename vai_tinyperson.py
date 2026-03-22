from ultralytics import YOLO

if "__main__" == __name__:
    model = YOLO(
        "C:/LJQ/ultralytics-main-ljq/runs/Person_result/train2/weights/best.pt")  # ai-

    results = model.val(
        data='person.yaml',
        batch=12,
        imgsz=640,
        device="0,1",
        workers=8,
        name='VisDrone_12change_neck',

        # conf=0.35,  # (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
        conf=0.001,
        iou=0.65  # (float) intersection over union (IoU) threshold for NMS
    )





