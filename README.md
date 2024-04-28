# Human Motion Recognition

![Human Motion Recognition](https://img.shields.io/badge/status-active-brightgreen.svg)

## Overview
Human Motion Recognition is a project aimed at detecting human presence and distinguishing between motion and stillness in live camera feeds or videos. It utilizes YOLOv3, a state-of-the-art deep neural network for object detection, along with the coco.names dataset for recognizing human objects.

## Features
- Real-time detection of human presence in live camera feeds.
- Differentiation between moving and static humans.
- Support for processing videos to detect human motion.
- Utilizes OpenCV's DNN module for loading and running the YOLOv3 pretrained model.

## Installation
1. Clone the repository:

    ```
    git clone https://github.com/your_username/human-motion-recognition.git
    ```

2. Navigate to the project directory:

    ```
    cd human-motion-recognition
    ```

3. Install dependencies:

    ```
    pip install -r requirements.txt
    ```

## Usage
1. Download the YOLOv3 weights file and place it in the project folder. [Click Here to Download](https://yolov3.weights)

2. Run the main script:

    ```
    python main4.py
    ```

3. Follow on-screen instructions for camera feed processing or provide the path to the video file for analysis.

## Additional Information
- YOLOv3 is a highly accurate and efficient object detection model, capable of real-time processing.
- The coco.names dataset provides the labels for various objects that the model can detect, including humans.
- OpenCV's DNN module is used for loading the pretrained YOLOv3 model and performing inference on input frames.


## Contribution
Contributions are welcome! Feel free to open issues or pull requests for any improvements or features you'd like to add.

## Acknowledgements
- YOLOv3: [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
- COCO Dataset: [Common Objects in Context](https://cocodataset.org/)
- OpenCV: [Open Source Computer Vision Library](https://opencv.org/)

## Author
[deebaadithya](https://github.com/deebaadithya)

For any questions or support, please [open an issue](https://github.com/your_username/human-motion-recognition/issues).
