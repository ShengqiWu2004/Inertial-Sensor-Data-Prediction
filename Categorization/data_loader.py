import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

# --- Label Mapping ---
LABEL_MAPPING = {
    'falldown': 0,
    'shaking': 1,
    'downstair': 2, 
    'jogging': 3, 
    'walking': 4, 
    'upstair': 5,
}
CATEGORY_KEYS = ['falldown', 'shaking', 'downstair', 'jogging', 'walking', 'upstair']

def load_and_label_data(data_dir, window_size=None, stride=10):
    """
    Loads CSV files from data_dir, assigns labels based on filename keywords,
    and augments data by slicing each file with a sliding window.
    
    Args:
        data_dir (str): Directory containing CSV files.
        window_size (int): Number of rows per segment. If None, uses the minimum file length.
        stride (int): Number of time steps to slide the window.
    
    Returns:
        data_list (list): List of numpy arrays (each array is a segment).
        labels (list): Corresponding label for each segment.
    """
    data_list = []
    labels = []
    valid_files = []
    
    # First pass: read files and filter by label
    for file in os.listdir(data_dir):
        if file.startswith('.'):
            continue
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path, encoding='latin1')
        # Assume sensor data is all columns except the first
        sensor_data = df.iloc[:, 1:].values

        label = None
        for key in LABEL_MAPPING:
            if key in file.lower():
                label = LABEL_MAPPING[key]
                break
        if label is None:
            print(f"Warning: No label found for {file}. Skipping.")
            continue

        valid_files.append((sensor_data, label))
    
    # If window_size is not provided, use the minimum number of rows across files
    if window_size is None:
        window_size = min(sensor_data.shape[0] for sensor_data, _ in valid_files)
        print(f"No window_size specified. Using minimum file length: {window_size}")
    
    # Slide a window over each file
    for sensor_data, label in valid_files:
        L = sensor_data.shape[0]
        if L < window_size:
            print(f"File length {L} is less than window_size {window_size}. Skipping file.")
            continue
        for i in range(0, L - window_size + 1, stride):
            segment = sensor_data[i:i+window_size]
            data_list.append(segment)
            labels.append(label)
    return data_list, labels

def normalize_segments(segments, scaler=None):
    """
    Normalize segments using StandardScaler.
    
    Args:
        segments (np.array): Array of shape (num_segments, window_size, num_features)
        scaler (StandardScaler, optional): If provided, use it to transform the data.
    
    Returns:
        normalized (np.array): Normalized segments with the same shape.
        scaler (StandardScaler): Fitted scaler.
    """
    num_segments, window_size, num_features = segments.shape
    reshaped = segments.reshape(-1, num_features)
    if scaler is None:
        scaler = StandardScaler()
        normalized = scaler.fit_transform(reshaped)
    else:
        normalized = scaler.transform(reshaped)
    normalized = normalized.reshape(num_segments, window_size, num_features)
    return normalized, scaler

def train_val_test_split_data(segments, labels, test_size=0.2, val_size=0.2, random_state=42):
    """
    Splits data into training, validation, and test sets.
    
    Args:
        segments (np.array): Data array.
        labels (np.array): Label array.
        test_size (float): Fraction of data to reserve for testing.
        val_size (float): Fraction of remaining data to reserve for validation.
        random_state (int): Seed for reproducibility.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        segments, labels, test_size=test_size, random_state=random_state, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state, shuffle=True
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

class SensorDataset(Dataset):
    """
    A PyTorch Dataset for sensor data.
    """
    def __init__(self, segments, labels, one_hot=False, num_classes=None):
        self.segments = torch.tensor(segments, dtype=torch.float32)
        if one_hot:
            if num_classes is None:
                raise ValueError("num_classes must be provided when one_hot is True")
            one_hot_labels = np.eye(num_classes)[labels]
            self.labels = torch.tensor(one_hot_labels, dtype=torch.float32)
        else:
            self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        return self.segments[idx], self.labels[idx]
