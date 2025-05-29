![Visitor Count](https://profile-counter.glitch.me/vyasdeepti/count.svg)

# Personal Protective Equipment (PPE) Object Detection using YOLOv11

This project provides an advanced Personal Protective Equipment (PPE) detection system using the YOLOv11 deep learning architecture. The objective is to automate the process of identifying whether individuals in images or video streams are wearing essential PPE items such as helmets, vests, gloves, and masks. The system is designed for real-time applications and can be deployed in industrial, construction, healthcare, or any environment where safety compliance is critical.

## Objectives

- **Automate PPE Monitoring:** Reduce manual supervision by automatically detecting PPE compliance in real-time.
- **Enhance Workplace Safety:** Improve safety standards by providing instant alerts when PPE is missing.
- **Flexible Integration:** Allow users to train or fine-tune on custom datasets and integrate the model into varied workflows.
- **Scalability:** Support for multiple input types (images, video files, live camera), making the solution adaptable to different environments.

Input image: 

![00020_jpg rf abe4d0d5d5829754ec268692af2e7307](https://github.com/user-attachments/assets/cd082164-684a-4d9a-8033-8f20bf70b652)

Output image: 

![00020_jpg rf abe4d0d5d5829754ec268692af2e7307](https://github.com/user-attachments/assets/ca4fd7e2-d4ca-406a-8ea8-097a1fffa67b)

 
## Key Features

- **YOLOv11 Model Integration:** Leverages the state-of-the-art YOLOv11 deep learning model for real-time object detection.
- **Custom PPE Dataset Support:** Easily train or fine-tune the model on your own annotated PPE dataset.
- **Multi-Class Detection:** Capable of detecting multiple PPE types simultaneously (helmets, vests, gloves, masks, etc.).
- **Real-Time Processing:** Optimized for fast inference on images, video files, or live camera streams.
- **Visualization:** Draws bounding boxes and labels on detected items in output images or video.
- **Configurable Thresholds:** Adjust detection confidence and non-max suppression thresholds for your application needs.

## Example Use Cases

- Monitoring compliance in construction sites.
- Automated safety checks in factories or warehouses.
- Ensuring mask/gear usage in healthcare facilities.
- Real-time alerts for missing PPE in hazardous zones.

## Project Structure

```
.
├── data/                   # Training, validation, and test datasets
├── models/                 # Pretrained or custom-trained YOLOv11 models
├── utils/                  # Utility scripts for preprocessing, visualization, etc.
├── detect.py               # Script for running detection on images/videos/streams
├── train.py                # Script for training or fine-tuning YOLOv11 on custom dataset
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── ...
```

Here’s a detailed explanation of each cell from the notebook python_code.ipynb in the PPE-Object-Detection-using-YOLO11 repo:

---

### **Cell 1**
```python
!pip install tensorflow
```
**Purpose:**  
Installs TensorFlow, a popular deep learning library. This is necessary for running or training deep learning models, though in this project, TensorFlow is not directly used (YOLO and Ultralytics use PyTorch as backend).


### **Cell 2**
```python
pip install tensorflow-gpu
```
**Purpose:**  
Installs the GPU version of TensorFlow to accelerate computations using your GPU (if available). Again, for YOLO and Ultralytics, PyTorch is more relevant.


### **Cell 3 (optional)**
```python
pip install nvidia-cuda-runtime-cu12
```
**Purpose:**  
Installs NVIDIA CUDA runtime version 12, which is a GPU acceleration library. Required for running deep learning models on NVIDIA GPUs.


### **Cell 4**
```python
!pip install ultralytics
from ultralytics import YOLO
createdmodel=YOLO("yolo11n.pt")
results= createdmodel.train(data="D:/YOLO-PPE-OBJECT DETECTION/dataset/data.yaml",epochs=8,imgsz=640)
```
**Purpose:**  
1. **Install Ultralytics:** This package provides the YOLO (You Only Look Once) object detection framework.
2. **Import YOLO:** Loads the YOLO class for model operations.
3. **Load Model:** `createdmodel=YOLO("yolo11n.pt")` loads the YOLOv11n (nano) model weights.
4. **Train Model:** Trains the model on your custom dataset (`data.yaml` describes classes and data paths) for 8 epochs, with image size 640.

**Output:**  
- Shows training progress, loss values, validation metrics, and final model saving.
- Detection categories in this context likely include: Person, Gloves, Hard_hat, Mask, Safety_boots, Vest.


### **Cell 5**
```python
#USING PRETRAINED MODEL
model_test=YOLO("runs/detect/train2/weights/best.pt")
results=model_test("test/images", save=True, imgsz=320, conf=0.7)
results[0].show()
```
**Purpose:**  
1. **Load Best Model:** Loads the best model weights from training.
2. **Run Inference:** Runs detection on all images in `test/images` with an image size of 320x320, saving results, and using a confidence threshold of 0.7.
3. **Show Results:** Displays the result for the first test image.

**Output:**  
- For each image, prints detections (e.g., “1 Person, 1 Vest”), inference time, and saves the visual result (annotated image).


### **Cell 6** 
```python
#USING PRETRAINED MODEL
model_test=YOLO("runs/detect/train2/weights/best.pt")
results=model_test("test/images", save=True, imgsz=320, conf=0.7)
results[0].show()
```
**Purpose:**  
This cell is again a repeat of the previous two. It runs inference on the test set and visualizes the predictions.

---

#### **Summary**

- **First 3 cells:** Install dependencies (some are not directly needed for YOLO/Ultralytics).
- **Cell 4:** Installs and uses Ultralytics YOLO, loads model, and trains on your dataset.
- **Cells 5-7:** Load the best model weights after training and run detection on test images, saving and displaying the results.



## Output Matrix

![confusion_matrix_normalized](https://github.com/user-attachments/assets/6b256d9d-8de1-4ba7-bc4a-464d67062be5)
![labels](https://github.com/user-attachments/assets/8c630f00-33a1-48a7-8d70-6bf31d66107f)
![results](https://github.com/user-attachments/assets/3b3aebe7-eaba-4262-85c6-c7a250dac059)
![PR_curve](https://github.com/user-attachments/assets/e6f1837f-f00b-492b-bbaa-d37623bc9090)


## References

- [YOLOv11 Paper/Repository](https://github.com/ultralytics/yolov11)  
- [YOLO Format Annotation Guide](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html) for pre-annotated PPE samples


















              
         
   
