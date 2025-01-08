# DU Parking Services Detector

This project is designed to detect parking services at DU.

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The DU Parking Services Detector is a tool developed to assist in the detection of parking services at DU. The primary goal is to detect the parking enforcement Toyota Prius.

## Getting Started

### Prerequisites
- Python (version 3.x)
- Required Python packages (listed in `requirements.txt`)

### Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/Willebrew/DU-Parking-Services-Detector.git
    ```
2. Navigate to the project directory:
    ```sh
    cd DU-Parking-Services-Detector
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
To use the DU Parking Services Detector, follow these steps:
1. Ensure all prerequisites are installed.
2. Run the live inference script:
    ```sh
    python inference_live.py
    ```

## Training the Model
The system uses a ResNet-18 model for binary classification of parking enforcement vehicles. You can train the model using the provided `train.py` script and a labeled dataset.

### Dataset Preparation
1. Use [Label Studio](https://labelstud.io/) to annotate your dataset. Label parking enforcement vehicles and export the annotations in COCO format.
2. Place your annotated dataset in the `Dataset/` directory and ensure your annotations file is named `result.json`.

### Training Steps
1. Ensure the training dataset is in the correct location (`Dataset/`) and formatted properly in COCO format.
2. Run the `train.py` script to start the training process:
    ```sh
    python train.py
    ```
3. The training process includes:
    - Loading the dataset with transformations for data augmentation.
    - Training a ResNet-18 model for binary classification.
    - Saving the trained model as `parking_vehicle_model.pth`.

During training, you will see metrics such as loss and accuracy per epoch displayed in the terminal.

### Model Details
- The final layer of ResNet-18 is modified for binary classification.
- Binary cross-entropy loss (`BCEWithLogitsLoss`) is used.
- The optimizer is Adam with a learning rate of 0.0001, and a step learning rate scheduler is applied.

The trained model can then be used in the live inference script for detection.

## Contributing
Contributions are welcome! Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch:
    ```sh
    git checkout -b feature-branch
    ```
3. Make your changes and commit them:
    ```sh
    git commit -m 'Add some feature'
    ```
4. Push to the branch:
    ```sh
    git push origin feature-branch
    ```
5. Open a Pull Request.

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
