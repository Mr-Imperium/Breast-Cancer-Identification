# Breast Cancer Histopathology Image Classification

## Project Overview
This project develops a deep learning model for classifying breast cancer histopathology images using a modified ResNet18 neural network. The application can predict whether a given microscopic image indicates cancerous or non-cancerous breast tissue.

## Key Features
- Binary classification of breast cancer histopathology images
- Custom ResNet18 architecture with additional layers
- Balanced class weights to handle dataset imbalance
- Streamlit web application for interactive predictions

## Technical Stack
- Python
- PyTorch
- Scikit-learn
- Streamlit
- Matplotlib

## Model Details
- Architecture: Modified ResNet18
- Input Size: 50x50 pixel images
- Preprocessing: Normalization, Random Flips
- Training Techniques:
  * Balanced class weighting
  * Cyclic learning rate
  * Xavier weight initialization

## Dataset
- Source: Breast Histopathology Images
- Classes: Cancerous vs Non-Cancerous
- Preprocessing: Coordinate extraction, patient-based splitting

## Model Performance
- Accuracy: Tracked during training
- Loss visualization provided
- Robust validation across patient-based splits

## Deployment
- Streamlit web application
- Easy-to-use interface
- Supports image upload and instant prediction

## Installation
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Run Streamlit app: `streamlit run app.py`

## Future Improvements
- Increase model complexity
- Augment training dataset
- Implement more advanced preprocessing techniques

## License
[MIT]
