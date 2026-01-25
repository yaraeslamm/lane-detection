# src/run_image.py
import os
import cv2
from inference import RoadSegmentationModel
from lane_postprocess import LanePostProcessor

MODEL_PATH = "models/road_model_industryV1_myDATASET.keras"
INPUT_DIR = "data/images"
OUTPUT_DIR = "outputs/images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

segmenter = RoadSegmentationModel(MODEL_PATH)
lane_processor = LanePostProcessor()

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(INPUT_DIR, fname)
    image = cv2.imread(img_path)
    if image is None:
        continue

    class_mask = segmenter.predict(image)

    # segmentation visualization (optional but GOOD for demo)
    color_mask = image.copy()
    color_mask[class_mask == 1] = (255, 0, 0)   # road
    color_mask[class_mask == 2] = (0, 255, 0)   # lane
    image_vis = cv2.addWeighted(image, 0.6, color_mask, 0.4, 0)

    result = lane_processor.process(image_vis, class_mask)

    out_path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(out_path, result)

    print(f"Saved: {out_path}")

print("Image inference completed")
