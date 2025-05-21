from ultralytics import YOLO

# Load a pretrained model
# model = YOLO("weights/lsb_telecom_v8.pt")
# model.train(
#     data="LSB-telcom-v9yolov11/data.yaml",
#     epochs=10,
#     imgsz=640,
#     batch=16,
#     name="LSB_telecom_model_v9_fine_tune_freeze10",
#     workers=0,
#     freeze=10,
#     amp=False,
# )

# model = YOLO("runs/detect/LSB_telecom_model_v9_fine_tune_freeze10/weights/best.pt")
# model.train(
#     data="LSB-telcom-v9yolov11/data.yaml",
#     epochs=40,
#     imgsz=640,
#     batch=16,
#     name="LSB_telecom_model_v9_fine_tune_unfreeze_all",
#     workers=0,
#     optimizer="AdamW",  # or "SGD", "Adam", … just not "auto"
#     lr0=1e-4,  # now it will be honoured
#     amp=False,
#     patience=20,
# )


# model = YOLO("runs/detect/LSB_receptacle_v1_from_yolov11s/weights/best.pt")
# model.train(
#     data="LSB_receptaclev2_fine_tune_yolov11s/data.yaml",
#     epochs=10,
#     imgsz=640,
#     batch=12,
#     name="LSB_receptacle_model_yolo11s_v1_fine_tune_freeze10",
#     workers=0,
#     freeze=10,
#     amp=False,
# )

model = YOLO(
    "runs/detect/LSB_receptacle_model_yolo11s_v1_fine_tune_freeze10/weights/best.pt"
)
model.train(
    data="LSB_receptaclev2_fine_tune_yolov11s/data.yaml",
    epochs=30,
    imgsz=640,
    batch=10,
    name="LSB_receptacle_model_yolo11s_v1_fine_tune_unfreeze_all",
    workers=0,
    # optimizer="AdamW",  # or "SGD", "Adam", … just not "auto"
    # lr0=1e-4,  # now it will be honoured
    amp=False,
    patience=20,
)


# model = YOLO("runs/detect/LSB_receptacle_v1_from_yolov11s/weights/best.pt")
# model.train(
#     data="LSB_receptaclev6_fine_tune_yolov11/data.yaml",
#     epochs=100,
#     imgsz=640,
#     batch=10,
#     name="LSB_receptacle_v1_from_yolov11s",
#     workers=0,
#     # optimizer="AdamW",  # or "SGD", "Adam", … just not "auto"
#     # lr0=1e-4,  # now it will be honoured
#     amp=False,
#     patience=40,
#     resume=True,
# )
