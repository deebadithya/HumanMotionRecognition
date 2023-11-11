import cv2
import numpy as np

# Load the pre-trained YOLOv3 model and configuration file
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load COCO class names (for object class labeling)
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Set the model's input size
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Initialize the video capture
cap = cv2.VideoCapture(0)  # Replace 'video.mp4' with your video source or use 0 for a live camera feed

prev_boxes = []  # Store the previous frame's detected person boxes

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    net.setInput(blob)
    
    layer_names = net.getUnconnectedOutLayersNames()
    outs = net.forward(layer_names)
    
    class_ids = []
    confidences = []
    boxes = []
    
    # Process the model's output to get detected objects
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    # Apply non-maximum suppression to filter out overlapping detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            
            if label == 'person':
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Determine if the person is moving or idle
                for prev_box in prev_boxes:
                    prev_x, prev_y, prev_w, prev_h = prev_box
                    if (
                        x > prev_x - 7 and x < prev_x + 7 and
                        y > prev_y - 7 and y < prev_y + 7
                    ):
                        movement_label = "Idle"
                        break
                else:
                    movement_label = "Moving"
                
                # Display the movement label
                cv2.putText(frame, movement_label, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    prev_boxes = boxes  # Update the previous frame's detected person boxes
    
    # Display the output frame
    cv2.imshow('Human Tracking', frame)
    
    # Exit the program with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
