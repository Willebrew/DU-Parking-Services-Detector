"""
This script performs live inference using a pre-trained ResNet-18 model for binary
classification of parking vehicles.

Modules:
    - torch: PyTorch library for deep learning.
    - torchvision: PyTorch library for vision-related tasks.
    - cv2: OpenCV library for real-time computer vision.

Functions:
    - predict_object(frame, threshold=0.5): Predicts the presence of a vehicle in the given frame.

Usage:
    Run the script to start the webcam feed and perform live inference.
    Press 'q' to quit the application.
"""
import torch
from torchvision.models import resnet18
import cv2
from model_utils import get_transform

# Define the image transformation pipeline
transform = get_transform()

# Load ResNet-18
model = resnet18()
# Modify the last layer for binary classification
model.fc = torch.nn.Linear(model.fc.in_features, 1)

# Load the model weights
model.load_state_dict(torch.load('parking_vehicle_model.pth'))
model.eval()

def predict_object(image_frame, threshold=0.5):
    """
    Predicts whether an object in the frame is a campus parking service vehicle.
    :param image_frame:
    :param threshold:
    :return:
    """
    image_tensor = transform(image_frame).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        live_prediction = torch.sigmoid(output).item()

    return live_prediction, prediction > threshold

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    prediction, detected = predict_object(frame, threshold=0.5)

    if detected:
        cv2.putText(frame, f"CAMPUS PARKING SERVICES DETECTED: {prediction:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"NO THREATS DETECTED: {prediction:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
