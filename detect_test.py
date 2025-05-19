from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("weights/last.pt")
img = cv2.imreadmulti
# Run batched inference on a list of images
img_paths = ["page_0001.png", "page_0002.png"]
results = model(img_paths)  # return a list of Results objects
images = []

for path in img_paths:
    img = cv2.imread(path)
    if img is not None:
        images.append(img)
    else:
        print(f"无法加载图片: {path}")
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()
# PAD = 64
# TILE = 512
# dets = []
# for r in results:
#     det = []
#     for box in r.boxes:
#         bx1, by1, bx2, by2 = box.xyxy[0].tolist()
#         cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
#         # keep only if centre in inner TILE × TILE
#         if PAD <= cx <= PAD + TILE and PAD <= cy <= PAD + TILE:
#             det.append(
#                 dict(
#                     label=r.names[int(box.cls)],
#                     confidence=box.conf[0].item(),
#                     x1=bx1,
#                     y1=by1,
#                     x2=bx2,
#                     y2=by2,
#                 )
#             )
#     dets.append(det)

# for i in range(len(dets)):
#     for det in dets[i]:
#         x1 = int(det["x1"])
#         y1 = int(det["y1"])
#         x2 = int(det["x2"])
#         y2 = int(det["y2"])
#         cv2.rectangle(images[i], (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.imwrite("detection_result" + str(i) + ".png", images[i])
