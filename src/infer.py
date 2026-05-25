import os
import json
from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
IMAGE_SOURCE = os.getenv("IMAGE_SOURCE", "dataset/valid/images")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "96"))
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.25"))
DEVICE = os.getenv("DEVICE", "cpu")
OUTPUT_JSON = Path(os.getenv("OUTPUT_JSON", "inference_results.json"))


def main():
    print("Git Model Package Inference")
    print("===========================")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Image Source: {IMAGE_SOURCE}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Confidence: {CONFIDENCE}")
    print(f"Device: {DEVICE}")

    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=IMAGE_SOURCE,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device=DEVICE,
        save=False,
        verbose=False,
    )

    output = []

    for result in results:
        image_result = {
            "image_path": str(result.path),
            "detections": []
        }

        names = result.names

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                detected_class = names.get(class_id, str(class_id))
                confidence = float(box.conf[0].item())
                xyxy = [float(x) for x in box.xyxy[0].tolist()]

                image_result["detections"].append({
                    "detected_class": detected_class,
                    "confidence": confidence,
                    "box_xyxy": xyxy
                })

        output.append(image_result)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"Inference Results gespeichert unter: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
