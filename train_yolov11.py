from ultralytics import YOLO

# Load a pretrained model
model = YOLO("weights/lsb_telecom_v8.pt")
model.train(
    data="LSB-telcom-v9yolov11/data.yaml",
    epochs=10,
    imgsz=640,
    batch=16,
    name="LSB_telecom_model_v9_fine_tune_freeze10",
    workers=0,
    optimizer="AdamW",  # or "SGD", "Adam", … just not "auto"
    lr0=1e-3,  # now it will be honoured
    momentum=0.937,  # (only matters for SGD/β1 for Adam*)
    freeze=10,
    amp=False,
    cache=True,
)

model = YOLO("runs/detect/LSB_telecom_model_v9_fine_tune_freeze10/weights/best.pt")
model.train(
    data="LSB-telcom-v9yolov11/data.yaml",
    epochs=40,
    imgsz=640,
    batch=16,
    name="LSB_telecom_model_v9_fine_tune_unfreeze_all",
    workers=0,
    optimizer="AdamW",  # or "SGD", "Adam", … just not "auto"
    lr0=1e-4,  # now it will be honoured
    momentum=0.937,  # (only matters for SGD/β1 for Adam*)
    amp=False,
    cache=True,
    patience=10,
)
