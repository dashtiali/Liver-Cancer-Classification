# %% [markdown]
# # Experiment with CNN
# ## Settings
# 1. Whole liver - All HU
# 2. Largest 15 slices
# 3. Soft voting


# %%
import os

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import ParameterGrid, train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# %% Setting the random state for reproducibility
RS = 123

# %% [markdown]
# ### Helper functions

# %%


def apply_windowing(ct_scan, win_center=40, win_width=350):
    # Calculate the lower and upper bounds of the window
    lower_bound = win_center - (win_width / 2)
    upper_bound = win_center + (win_width / 2)

    # Apply windowing
    windowed_scan = np.clip(ct_scan, lower_bound, upper_bound)

    # Normalize the windowed scan to be in the range [0, 1]
    windowed_scan = (windowed_scan - lower_bound) / (upper_bound - lower_bound)

    return windowed_scan

# %% [markdown]
# ### CNN Setup and Training (xception)

# %% [markdown]
# #### Data Setup


# %%
# Define paths and variables
prepared_data_dir = 'prepared_data'
X_train = np.load(os.path.join(prepared_data_dir, 'X_train.npy'))
X_test = np.load(os.path.join(prepared_data_dir, 'X_test.npy'))
y_train = np.load(os.path.join(prepared_data_dir, 'y_train.npy'))
y_test = np.load(os.path.join(prepared_data_dir, 'y_test.npy'))

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

# Apply windowing to both the training and testing data
X_train = np.array([apply_windowing(patient) for patient in X_train])
X_val = np.array([apply_windowing(patient) for patient in X_val])
X_test = np.array([apply_windowing(patient) for patient in X_test])

# Define custom dataset


class PatientDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data) * 15  # each patient has 15 slices

    def __getitem__(self, idx):
        patient_idx = idx // 15
        slice_idx = idx % 15
        x = self.data[patient_idx][slice_idx]
        x = Image.fromarray((x * 255).astype(np.uint8))
        x = x.convert("RGB")
        y = self.labels[patient_idx]

        if self.transform:
            x = self.transform(x)

        return x, y


# Define transformations for the network
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create datasets and dataloaders
train_dataset = PatientDataset(X_train, y_train, transform=transform)
val_dataset = PatientDataset(X_val, y_val, transform=transform)
test_dataset = PatientDataset(X_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %% [markdown]
# #### Training and Hyperparameter Tuning

# %%


def train_model(params):
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    num_epochs = params['num_epochs']
    patience = params['patience']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = timm.create_model('xception', pretrained=True, num_classes=1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs = inputs.to(device).float()
            labels = labels.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.sigmoid(outputs) > 0.5
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
                inputs = inputs.to(device).float()
                labels = labels.to(device).float().view(-1, 1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                preds = torch.sigmoid(outputs) > 0.5
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_model_xception.pth')
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print('Early stopping')
            break

        scheduler.step()

    return best_val_acc.item()


# %%
# Define hyperparameter grid
param_grid = {
    'batch_size': [16, 32],
    'learning_rate': [0.001, 0.0001],
    'weight_decay': [0, 0.0001],
    'num_epochs': [50],
    'patience': [5, 10]
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Perform grid search
best_acc = 0.0
best_params = None

for params in ParameterGrid(param_grid):
    print(f"Training with parameters: {params}")
    val_acc = train_model(params)
    print(f"Validation Accuracy: {val_acc}")

    if val_acc > best_acc:
        best_acc = val_acc
        best_params = params

print(f"Best Validation Accuracy: {best_acc}")
print(f"Best Hyperparameters: {best_params}")

# %% [markdown]
# #### Evaluation

# %%
# Evaluation


def evaluate_model(model, test_loader):
    model.eval()  # set model to evaluate mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device).float()
            labels = labels.to(device).float().view(-1, 1)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Group predictions by patient
    patient_preds = []
    patient_labels = []

    num_slices_per_patient = 15  # each patient has 15 slices

    for i in range(0, len(all_preds), num_slices_per_patient):
        patient_pred = np.mean(all_preds[i:i + num_slices_per_patient])
        patient_label = np.mean(all_labels[i:i + num_slices_per_patient])
        patient_preds.append(patient_pred)
        patient_labels.append(patient_label)

    patient_preds = np.array(patient_preds)
    patient_labels = np.array(patient_labels)

    # Convert patient labels and predictions to binary
    patient_labels_binary = patient_labels > 0.5
    patient_preds_binary = patient_preds > 0.5

    # Calculate all metrics
    accuracy = accuracy_score(patient_labels_binary, patient_preds_binary)
    auc = roc_auc_score(patient_labels_binary, patient_preds)
    recall = recall_score(patient_labels_binary, patient_preds_binary)
    precision = precision_score(patient_labels_binary, patient_preds_binary)
    f1 = f1_score(patient_labels_binary, patient_preds_binary)

    # Create metrics data frame
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'AUC', 'Recall', 'Precision', 'F1-score'],
        'Value': [accuracy, auc, recall, precision, f1]
    })

    print(metrics_df)


# Load the best model and perform evaluation on the test set
model = timm.create_model('xception', pretrained=True, num_classes=1)

model.load_state_dict(torch.load('best_model_xception.pth'))
model = model.to(device)
evaluate_model(model, test_loader)
# Load the best model and perform evaluation on the test set
model = timm.create_model('xception', pretrained=True, num_classes=1)

model.load_state_dict(torch.load('best_model_xception.pth'))
model = model.to(device)
evaluate_model(model, test_loader)
