import cv2
import numpy as np
import random

# Load object names from a text file
def load_object_names(file_path):
    with open(file_path, 'r') as file:
        object_names = file.read().splitlines()
    return object_names

# Perform object detection using YOLO
def detect_objects(frame, net, output_layers, classes):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Lists to store detection information
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id < len(classes):
                center_x, center_y, w, h = map(int, detection[:4] * [width, height, width, height])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    detected_objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            detected_objects.append((classes[class_ids[i]], confidences[i], (x, y, w, h), class_ids[i]))

    return detected_objects

# Generate a distinct color for each class
def generate_color_for_class(class_id):
    random.seed(class_id)
    color = [random.randint(0, 255) for _ in range(3)]
    return tuple(color)

# Save detection data to a text file
def save_detection_data(detections, file_path):
    with open(file_path, 'a') as file:  # Open file in append mode
        for obj, conf, bbox, class_id in detections:
            x, y, w, h = bbox
            line = f"Object: {obj}, Confidence: {conf:.2f}, BoundingBox: (x: {x}, y: {y}, w: {w}, h: {h}), ClassID: {class_id}\n"
            file.write(line)

# Main Function
def main():
    # Load YOLO model
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

    # Enable CUDA if available (GPU Acceleration)
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("CUDA enabled, using GPU for inference.")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        print("CUDA not available, using CPU for inference.")

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # Load object names
    classes = load_object_names("objects.txt")

    # Open camera
    cap = cv2.VideoCapture(0)

    # Text file for storing detection data
    detection_log_file = "detections_log.txt"

    # Clear any existing data in the log file at the start
    with open(detection_log_file, 'w') as file:
        file.write("Object Detection Log\n\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        detections = detect_objects(frame, net, output_layers, classes)

        # Save detections to the text file
        save_detection_data(detections, detection_log_file)

        # Display detections
        for obj, conf, bbox, class_id in detections:
            x, y, w, h = bbox
            label = f"{obj} ({conf*100:.2f}%)"
            color = generate_color_for_class(class_id)  # Generate distinct color for the object class

            # Draw bounding box and label with distinct color
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show frame
        cv2.imshow("Object Detection", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
