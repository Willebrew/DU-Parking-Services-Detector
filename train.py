"""
train.py

This script is used to train a ResNet-18 model for binary classification of parking enforcement
vehicles. The training data is sourced from a custom COCO dataset, and the model is trained
using various image transformations to improve generalization.

Classes:
    ParkingVehicleDataset: Custom dataset class for parking vehicle detection.

Functions:
    None

Dependencies:
    - torch
    - torchvision
    - tqdm
"""
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
from model_utils import get_transform

# Define the image transformation pipeline
transform = get_transform(train = True)


class ParkingVehicleDataset(CocoDetection):
    """
    Custom dataset class for parking vehicle detection, inheriting from CocoDetection.
    """

    def __getitem__(self, index):
        """
        Override the __getitem__ method to return image and binary label.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: (image, label) where label is 1 if a vehicle is present, else 0.
        """
        image, target = super().__getitem__(index)
        label = torch.tensor(1.0) if len(target) > 0 else torch.tensor(0.0)
        return image, label


if __name__ == '__main__':
    # Initialize the dataset and data loader
    train_dataset = ParkingVehicleDataset(root='Dataset',
                                          annFile='Dataset/result.json',
                                          transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    # Load the ResNet18 and modify the final layer for binary classification
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Change the output layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss function, optimizer, and learning rate scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    NUM_EPOCS = 100
    for epoch in range(NUM_EPOCS):
        model.train()
        RUNNING_LOSS = 0.0
        CORRECT = 0
        TOTAL = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCS}")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels.float().unsqueeze(1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running loss and accuracy
            RUNNING_LOSS += loss.item()
            predicted = torch.sigmoid(outputs) > 0.5
            TOTAL += labels.size(0)
            CORRECT += (predicted.squeeze() == labels).sum().item()

            progress_bar.set_postfix({'loss': f"{RUNNING_LOSS / len(train_loader):.4f}",
                                      'accuracy': f"{100 * CORRECT / TOTAL:.2f}%"})

        # Step the learning rate scheduler
        scheduler.step()

        print(f"Epoch {epoch + 1}/{RUNNING_LOSS}")
        print(f"Loss: {RUNNING_LOSS / len(train_loader):.4f}")
        print(f"Accuracy: {100 * CORRECT / TOTAL:.2f}%")
        print()

    torch.save(model.state_dict(), 'parking_vehicle_model.pth')
