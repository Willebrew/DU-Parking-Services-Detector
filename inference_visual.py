"""
inference_visual.py

This script is used to perform inference on images to detect campus parking service vehicles
using a pre-trained ResNet-18 model. The model is trained for binary classification and
expects images to be preprocessed with specific transformations.

Functions:
    predict_object: Predicts whether an object in the image is a campus parking service vehicle.

Dependencies:
    - torch
    - torchvision
    - PIL (Pillow)
"""
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load ResNet-18
model = resnet18()
# Change the output layer for binary classification
model.fc = torch.nn.Linear(model.fc.in_features, 1)

# Load the model weights
model.load_state_dict(torch.load('parking_vehicle_model.pth'))
model.eval()


def predict_object(image_path, threshold=0.5):
    """
    Predicts whether an object in the image is a campus parking service.

    Args:
        image_path (str): Path to the input image.
        threshold (float): Confidence threshold for detection.

    Returns:
        bool: True if the object is detected, False otherwise.
    """
    # Open and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output).item()

    if prediction > threshold:
        print("CAMPUS PARKING SERVICES DETECTED")
        print(f"Confidence: {prediction:.2f}")
    else:
        print("NO THREATS DETECTED")
        print(f"Confidence: {prediction:.2f}")

    return prediction > threshold

IMAGE = 'Dataset/images/dce226f9-IMG_6682.png'
detected = predict_object(IMAGE, threshold=0.5)
