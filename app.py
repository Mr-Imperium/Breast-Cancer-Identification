import streamlit as st
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms
import numpy as np

# Configuration
NUM_CLASSES = 2

class BreastCancerClassifier(nn.Module):
    def __init__(self):
        super(BreastCancerClassifier, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=False)
        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            
            nn.Linear(256, NUM_CLASSES)
        )
    
    def forward(self, x):
        return self.model(x)

def load_model(model_path):
    model = BreastCancerClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(model, image):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
        return predicted.item(), probabilities[0]

def main():
    st.title('Breast Cancer Histopathology Image Classifier')
    
    # Model loading
    try:
        model = load_model('breast_cancer_model.pth')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a breast histopathology image...", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Prepare image for prediction
        transformed_image = transform_image(image)
        
        # Make prediction
        prediction, probabilities = predict(model, transformed_image)
        
        # Display results
        st.subheader('Prediction Results')
        
        class_labels = ['Non-Cancerous', 'Cancerous']
        st.write(f"Predicted Class: **{class_labels[prediction]}**")
        
        # Show probabilities
        st.write("Confidence:")
        for i, prob in enumerate(probabilities):
            st.write(f"{class_labels[i]}: {prob.item()*100:.2f}%")

if __name__ == "__main__":
    main()