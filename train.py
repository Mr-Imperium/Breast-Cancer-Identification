import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from glob import glob
from skimage.io import imread
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
BATCH_SIZE = 32
NUM_CLASSES = 2
BASE_PATH = "../input/breast-histopathology-images/IDC_regular_ps50_idx5/"
OUTPUT_PATH = "/kaggle/working/"
MODEL_PATH = "../input/breastcancermodel/"

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

def extract_coords(df):
    """Extract x and y coordinates from image paths."""
    coord = df.path.str.rsplit("_", n=4, expand=True)
    coord = coord.drop([0, 1, 4], axis=1)
    coord = coord.rename({2: "x", 3: "y"}, axis=1)
    coord.loc[:, "x"] = coord.loc[:,"x"].str.replace("x", "", case=False).astype(int)
    coord.loc[:, "y"] = coord.loc[:,"y"].str.replace("y", "", case=False).astype(int)
    df.loc[:, "x"] = coord.x.values
    df.loc[:, "y"] = coord.y.values
    return df

def my_transform(key="train", plot=False):
    """Create data transformations for training and validation."""
    train_sequence = [transforms.Resize((50,50)),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip()]
    val_sequence = [transforms.Resize((50,50))]
    
    if not plot:
        train_sequence.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        val_sequence.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    data_transforms = {'train': transforms.Compose(train_sequence),
                       'val': transforms.Compose(val_sequence)}
    return data_transforms[key]

class BreastCancerDataset(Dataset):
    """Custom Dataset for Breast Cancer Image Classification."""
    def __init__(self, df, transform=None):
        self.states = df
        self.transform = transform
      
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        image_path = self.states.path.values[idx] 
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        target = int(self.states.target.values[idx]) if "target" in self.states.columns else None
            
        return {
            "image": image,
            "label": target,
            "patient_id": self.states.patient_id.values[idx],
            "x": self.states.x.values[idx],
            "y": self.states.y.values[idx]
        }

def prepare_data(base_path):
    """Prepare dataset from image files."""
    folder = os.listdir(base_path)
    total_images = sum(len(os.listdir(os.path.join(base_path, patient_id, str(c)))) 
                       for patient_id in folder for c in [0, 1])
    
    data = pd.DataFrame(index=np.arange(0, total_images), 
                        columns=["patient_id", "path", "target"])
    
    k = 0
    for patient_id in folder:
        patient_path = os.path.join(base_path, patient_id)
        for c in [0, 1]:
            class_path = os.path.join(patient_path, str(c))
            subfiles = os.listdir(class_path)
            for image_file in subfiles:
                data.iloc[k]["path"] = os.path.join(class_path, image_file)
                data.iloc[k]["target"] = c
                data.iloc[k]["patient_id"] = patient_id
                k += 1
    
    return data

def create_model():
    """Create and modify ResNet18 for binary classification."""
    model = torchvision.models.resnet18(pretrained=False)
    
    # Load pre-trained weights if available
    try:
        model.load_state_dict(torch.load("../input/pretrained-pytorch-models/resnet18-5c106cde.pth"))
    except FileNotFoundError:
        print("Pre-trained weights not found. Training from scratch.")
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
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
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    model.apply(init_weights)
    return model

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, 
                num_epochs=30, scheduler=None):
    """Train the model with specified parameters."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    loss_history = {"train": [], "dev": [], "test": []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'dev']:
            model.train() if phase == 'train' else model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            with torch.set_grad_enabled(phase == 'train'):
                for batch in dataloaders[phase]:
                    inputs = batch['image'].to(device)
                    labels = batch['label'].to(device)
                    
                    optimizer.zero_grad()
                    
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if scheduler:
                            scheduler.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            loss_history[phase].append(epoch_loss)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'dev' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        
        print()
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_history

def main():
    # Prepare data
    data = prepare_data(BASE_PATH)
    data['target'] = data.target.astype(int)
    
    # Split data by patient
    patients = data.patient_id.unique()
    train_ids, sub_test_ids = train_test_split(patients, test_size=0.3, random_state=0)
    test_ids, dev_ids = train_test_split(sub_test_ids, test_size=0.5, random_state=0)
    
    # Create dataframes
    train_df = data[data.patient_id.isin(train_ids)].copy()
    test_df = data[data.patient_id.isin(test_ids)].copy()
    dev_df = data[data.patient_id.isin(dev_ids)].copy()
    
    # Extract coordinates
    train_df = extract_coords(train_df)
    test_df = extract_coords(test_df)
    dev_df = extract_coords(dev_df)
    
    # Create datasets and dataloaders
    train_dataset = BreastCancerDataset(train_df, transform=my_transform(key="train"))
    dev_dataset = BreastCancerDataset(dev_df, transform=my_transform(key="val"))
    test_dataset = BreastCancerDataset(test_df, transform=my_transform(key="val"))
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    dataloaders = {
        "train": train_dataloader, 
        "dev": dev_dataloader, 
        "test": test_dataloader
    }
    
    dataset_sizes = {
        "train": len(train_dataset), 
        "dev": len(dev_dataset), 
        "test": len(test_dataset)
    }
    
    # Define device at the top of the main function or before creating the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Compute class weights
    weights = compute_class_weight(
    class_weight="balanced", 
    classes=data.target.unique(), 
    y=train_df.target.values
    )
    class_weights = torch.FloatTensor(weights).to(device)

    # Create model and criterion
    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.fc.parameters(), lr=0.01)
    
    # Optional: Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=1e-6, 
        max_lr=0.006, 
        step_size_up=len(train_dataloader)
    )
    
    # Train model
    trained_model, loss_history = train_model(
        model, dataloaders, dataset_sizes, 
        criterion, optimizer, 
        num_epochs=30, 
        scheduler=scheduler
    )
    
    # Save model
    torch.save(trained_model.state_dict(), os.path.join(OUTPUT_PATH, 'breast_cancer_model.pth'))
    
    # Optional: Plot and save loss history
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history['train'], label='Train Loss')
    plt.plot(loss_history['dev'], label='Dev Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_PATH, 'loss_history.png'))
    plt.close()

if __name__ == "__main__":
    main()
