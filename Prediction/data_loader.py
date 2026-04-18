import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from collections import defaultdict

# These are the normal categories that will be balanced equally.
CATEGORY_KEYS = ['falldown','shaking','downstair', 'jogging', 'walking', 'upstair']


def extract_subject_id(filename):
    """Extract subject ID (e.g., 'S1') from filename like 'S1_t1_downstair_falldown.csv'."""
    match = re.match(r'(S\d+)_', filename, re.IGNORECASE)
    return match.group(1).upper() if match else None


def get_all_subjects(data_dir):
    """Return sorted list of all unique subject IDs found in data_dir."""
    subjects = set()
    for f in os.listdir(data_dir):
        if f.startswith('.'):
            continue
        sid = extract_subject_id(f)
        if sid:
            subjects.add(sid)
    return sorted(subjects)


def load_and_segment_data(data_dir, window_size, predict_size, balance_config,
                          include_subjects=None, balance=True, step_ratio=0.7):
    """
    Load CSV files from data_dir and segment them into input/output pairs.

    Files are expected to follow the naming convention: S{n}_t{n}_{category}[_{anomaly}].csv
    where Sn is the nth subject and tn is the nth trial.

    Args:
      include_subjects: if provided (e.g. ['S1','S3']), only load files from those subjects.
      balance: if True, under-sample each category to the minimum count across categories.
                if False, return all segments without balancing (used for test evaluation).

    Returns:
      X: np.array of shape (num_segments, window_size, 6)
      y: np.array of shape (num_segments, predict_size, 6)
      labels: np.array of category strings corresponding to each segment.
    """
    segments_by_cat = defaultdict(list)
    targets_by_cat = defaultdict(list)

    step_size = int(step_ratio * window_size)
    seg_length = window_size + predict_size

    # Process each file in the directory
    for file in os.listdir(data_dir):
        if file.startswith('.'):
            continue
        # Filter by subject for LOSO cross-validation
        if include_subjects is not None:
            sid = extract_subject_id(file)
            if sid not in include_subjects:
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
        # if SPECIAL_CATEGORY in file_lower:
        #     category = SPECIAL_CATEGORY
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
    
    balanced_segments = []
    balanced_targets = []
    balanced_labels = []

    if balance:
        # Under-sample each category to the minimum count across categories.
        normal_counts = [segments_by_cat[cat].shape[0] for cat in CATEGORY_KEYS if cat in segments_by_cat]
        min_normal = min(normal_counts) if normal_counts else 0
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
    else:
        # Return all segments unmodified (used for held-out test subjects in LOSO).
        for cat in CATEGORY_KEYS:
            if cat in segments_by_cat:
                balanced_segments.append(segments_by_cat[cat])
                balanced_targets.append(targets_by_cat[cat])
                balanced_labels += [cat] * segments_by_cat[cat].shape[0]

    # Concatenate all segments and shuffle.
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
        self.labels = labels

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.segments[idx], self.targets[idx], self.labels[idx]
        else:
            return self.segments[idx], self.targets[idx]



def train_val_split(X, y, labels, val_size=0.2, random_state=42):
    """
    Split data into train and validation sets (stratified by label).
    Used within each LOSO fold to reserve 20% of training subjects' data for early stopping.

    Returns: X_train, X_val, y_train, y_val, labels_train, labels_val
    """
    X_train, X_val, y_train, y_val, labels_train, labels_val = train_test_split(
        X, y, labels, test_size=val_size, random_state=random_state, stratify=labels
    )
    return X_train, X_val, y_train, y_val, labels_train, labels_val


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
    
    return X_train, X_val, X_test, y_train, y_val, y_test, labels_train, labels_val, labels_test
