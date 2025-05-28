
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


## Output Matrix

![confusion_matrix_normalized](https://github.com/user-attachments/assets/6b256d9d-8de1-4ba7-bc4a-464d67062be5)
![labels](https://github.com/user-attachments/assets/8c630f00-33a1-48a7-8d70-6bf31d66107f)
![results](https://github.com/user-attachments/assets/3b3aebe7-eaba-4262-85c6-c7a250dac059)
![PR_curve](https://github.com/user-attachments/assets/e6f1837f-f00b-492b-bbaa-d37623bc9090)








## References

- [YOLOv11 Paper/Repository](https://github.com/ultralytics/yolov11)  
- [YOLO Format Annotation Guide](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html) for pre-annotated PPE samples


















              
         
   
