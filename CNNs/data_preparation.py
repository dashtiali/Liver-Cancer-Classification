# CNN data preparation
# Settings:
# 1. Whole liver - All HU
# 2. Largest 15 slices


import os
import numpy as np
from glob import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split


RS = 123

# Patient Data Preparation

# Define paths and variables
main_dir = r'HCC_ICC_dataset_numpy_arrays'
class_mapping = {'ICC': 0, 'HCC': 1}
data = defaultdict(list)
labels = defaultdict(list)

# Helper function to extract patient ID from file name
def get_patient_id(file_name):
    return file_name.split('_slice')[0]

# Load data
for class_name, class_label in class_mapping.items():
    class_dir = os.path.join(main_dir, class_name)
    files = glob(os.path.join(class_dir, '*.npy'))
    for file_path in files:
        patient_id = get_patient_id(os.path.basename(file_path))
        array_data = np.load(file_path)
        data[patient_id].append(array_data)
        labels[patient_id].append(class_label)

# Verify that each patient has 15 data points
for patient_id in data.keys():
    if len(data[patient_id]) != 15:
        print(f"Patient {patient_id} has {len(data[patient_id])} data points instead of 15.")

# Create the dataset for the CNN
X = []
y = []

for patient_id in data.keys():
    X.append(np.stack(data[patient_id]))  # stack the 15 slices
    y.append(labels[patient_id][0])  # assign label

X = np.array(X)
y = np.array(y)

# Calculate the test ratio (Round down)
test_ratio = int(X.shape[0] * 0.2) / X.shape[0]

# Dataset splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, stratify=y, random_state=RS)

print("Data prepared for CNN:")
print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Testing set: {X_test.shape}, {y_test.shape}")

# Path to save the prepared data
prepared_data_dir = 'prepared_data'
os.makedirs(prepared_data_dir, exist_ok=True)

# Export the prepared dataset
np.save(os.path.join(prepared_data_dir, 'X_train.npy'), X_train)
np.save(os.path.join(prepared_data_dir, 'X_test.npy'), X_test)
np.save(os.path.join(prepared_data_dir, 'y_train.npy'), y_train)
np.save(os.path.join(prepared_data_dir, 'y_test.npy'), y_test)

print(f"Prepared data saved in '{prepared_data_dir}' directory.")
