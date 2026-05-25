import os
from pathlib import Path

from ultralytics import YOLO


MODEL_ID = os.getenv("MODEL_ID", "sustainable-ai-model-v1")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0")
LOCATION_NAME = os.getenv("LOCATION_NAME", "unknown")
REGION_CODE = os.getenv("REGION_CODE", "unknown")

DATASET_YAML = os.getenv("DATASET_YAML", "dataset/data.yaml")
BASE_MODEL = os.getenv("BASE_MODEL", "yolov8n.pt")
EPOCHS = int(os.getenv("EPOCHS", "1"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "96"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
DEVICE = os.getenv("DEVICE", "cpu")

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "runs/train"))


def main():
    print("Git Model Package Training")
    print("==========================")
    print(f"Model ID: {MODEL_ID}")
    print(f"Model Version: {MODEL_VERSION}")
    print(f"Location: {LOCATION_NAME}")
    print(f"Region Code: {REGION_CODE}")
    print(f"Dataset YAML: {DATASET_YAML}")
    print(f"Base Model: {BASE_MODEL}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")

    model = YOLO(BASE_MODEL)

    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(OUTPUT_DIR),
        name=f"{REGION_CODE}_{LOCATION_NAME}_git_package_train",
        save=True,
        plots=False,
        verbose=True,
    )


if __name__ == "__main__":
    main()
