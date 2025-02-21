import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from collections import defaultdict

# These are the normal categories that will be balanced equally.
CATEGORY_KEYS = ['downstair', 'jogging', 'walking', 'upstair']
# This is the special category that will be sampled with a controlled ratio.
SPECIAL_CATEGORY = 'falldown'


def load_and_segment_data(data_dir, window_size, predict_size, balance_config, mode="train"):
    """
    Load CSV files from data_dir and segment them into input/output pairs.
    
    Assumptions:
      - Each CSV file contains 7 columns. The first column is ignored (e.g., timestamp)
        and the remaining 6 columns are sensor readings.
      - The file name should contain one of the keywords in CATEGORY_KEYS or SPECIAL_CATEGORY.
      
    Segmentation:
      - For each file, slide a window of length (window_size + predict_size) with a step
        size equal to 80% of window_size.
      - For each window, the first window_size rows are used as X (input) and the following
        predict_size rows are used as y (target).
      
    Balancing:
      - For each category in CATEGORY_KEYS, only keep as many segments as the minimum count 
        across these categories.
      - For the SPECIAL_CATEGORY ("falldown"), keep only a fraction of the segments as given 
        by balance_config['falldown_ratio'].
    
    Returns:
      X: np.array of shape (num_segments, window_size, 6)
      y: np.array of shape (num_segments, predict_size, 6)
      labels: np.array of category strings corresponding to each segment.
    """
    segments_by_cat = defaultdict(list)
    targets_by_cat = defaultdict(list)
    
    step_size = int(0.8 * window_size)
    seg_length = window_size + predict_size
    
    # Process each file in the directory
    for file in os.listdir(data_dir):
        if file.startswith('.'):
            continue
        file_path = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(file_path, encoding='latin1')
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
        
        # Sensor data: ignore the first column, use the rest (assumed to be 6 columns)
        sensor_data = df.iloc[:, 1:].values  # shape: (num_rows, 6)
        
        # Determine the category from the filename (case-insensitive)
        file_lower = file.lower()
        category = None
        if SPECIAL_CATEGORY in file_lower:
            category = SPECIAL_CATEGORY
        for key in CATEGORY_KEYS:
            if key in file_lower:
                category = key
                break
        if category is None:
            print(f"Warning: No recognized category in {file}. Skipping.")
            continue
        
        total_rows = sensor_data.shape[0]
        # Segment the file if there is enough data
        for i in range(0, total_rows - seg_length + 1, step_size):
            X_segment = sensor_data[i: i + window_size]
            y_segment = sensor_data[i + window_size: i + seg_length]
            segments_by_cat[category].append(X_segment)
            targets_by_cat[category].append(y_segment)
    
    # Convert lists to np.arrays per category
    for cat in segments_by_cat:
        segments_by_cat[cat] = np.array(segments_by_cat[cat])
        targets_by_cat[cat] = np.array(targets_by_cat[cat])
    
    # Balance the normal categories: pick the minimum count among them.
    normal_counts = [segments_by_cat[cat].shape[0] for cat in CATEGORY_KEYS if cat in segments_by_cat]
    if normal_counts:
        min_normal = min(normal_counts)
    else:
        min_normal = 0

    balanced_segments = []
    balanced_targets = []
    balanced_labels = []
    
    # For each normal category, randomly sample min_normal segments.
    for cat in CATEGORY_KEYS:
        if cat in segments_by_cat:
            seg = segments_by_cat[cat]
            targ = targets_by_cat[cat]
            if seg.shape[0] > min_normal:
                idx = np.random.choice(seg.shape[0], min_normal, replace=False)
                seg = seg[idx]
                targ = targ[idx]
            balanced_segments.append(seg)
            balanced_targets.append(targ)
            balanced_labels += [cat] * seg.shape[0]
    
    # For the special category, sample a fraction specified by falldown_ratio
    if SPECIAL_CATEGORY in segments_by_cat:
        seg = segments_by_cat[SPECIAL_CATEGORY]
        targ = targets_by_cat[SPECIAL_CATEGORY]
        ratio = balance_config.get('falldown_ratio', 1.0)  # default to 100% if not specified
        num_samples = int(seg.shape[0] * ratio)
        if num_samples < seg.shape[0]:
            idx = np.random.choice(seg.shape[0], num_samples, replace=False)
            seg = seg[idx]
            targ = targ[idx]
        balanced_segments.append(seg)
        balanced_targets.append(targ)
        balanced_labels += [SPECIAL_CATEGORY] * seg.shape[0]
    
    # Concatenate all the balanced segments and shuffle the overall order.
    if len(balanced_segments) == 0:
        raise ValueError("No data was loaded from directory " + data_dir)
    
    X = np.concatenate(balanced_segments, axis=0)
    y = np.concatenate(balanced_targets, axis=0)
    labels = np.array(balanced_labels)
    
    # (Optional) You could add a random shuffle here.
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    labels = labels[indices]
    
    return X, y, labels


def normalize_segments(segments, scaler=None):
    """
    Normalize segments using a StandardScaler. Each segment can be (window_size or predict_size, 6).
    The scaler is fitted on the flattened data.
    
    Returns:
      normalized segments, and the fitted scaler.
    """
    num_segments, seq_len, num_features = segments.shape
    reshaped = segments.reshape(-1, num_features)
    if scaler is None:
        scaler = StandardScaler()
        normalized = scaler.fit_transform(reshaped)
    else:
        normalized = scaler.transform(reshaped)
    normalized = normalized.reshape(num_segments, seq_len, num_features)
    return normalized, scaler


class SensorDataset(Dataset):
    """
    PyTorch Dataset for sensor data regression.
    Each sample is a tuple (X, y, label) where:
      - X is a tensor of shape (window_size, 6)
      - y is a tensor of shape (predict_size, 6)
      - label is the category (if provided)
    """
    def __init__(self, segments, targets, labels=None, normalize=True, scaler=None):
        if normalize:
            segments, self.scaler = normalize_segments(segments, scaler)
            targets, _ = normalize_segments(targets, scaler)
        else:
            self.scaler = scaler
        self.segments = torch.tensor(segments, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.labels = labels  # optional; can be None

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.segments[idx], self.targets[idx], self.labels[idx]
        else:
            return self.segments[idx], self.targets[idx]



def train_val_test_split(segments, targets, labels, test_size=0.2, val_size=0.4, random_state=42):
    """
    Split data into train, validation, and test sets, including labels.
    Returns:
      X_train, X_val, X_test, y_train, y_val, y_test, labels_train, labels_val, labels_test
    """
    from sklearn.model_selection import train_test_split
    # First, split out the test set with stratification on labels.
    X_train_val, X_test, y_train_val, y_test, labels_train_val, labels_test = train_test_split(
        segments, targets, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Now split train_val into train and validation sets.
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, labels_train, labels_val = train_test_split(
        X_train_val, y_train_val, labels_train_val, test_size=val_fraction, random_state=random_state, stratify=labels_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test #  , labels_train, labels_val, labels_test
