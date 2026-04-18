import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

# --- Label Mapping ---
# Labels are assigned by the FIRST matching keyword (in priority order).
# Files like 'S1_t1_downstair_falldown.csv' get label 'falldown' (anomalous class).
LABEL_MAPPING = {
    'falldown': 0,
    'shaking': 1,
    'downstair': 2,
    'jogging': 3,
    'walking': 4,
    'upstair': 5,
}
CATEGORY_KEYS = ['falldown', 'shaking', 'downstair', 'jogging', 'walking', 'upstair']


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


def balance_segments(segments, labels):
    """
    Under-sample each class to the count of the smallest class.
    Returns balanced segments and labels as numpy arrays.
    """
    segments = np.array(segments)
    labels = np.array(labels)
    unique_classes, counts = np.unique(labels, return_counts=True)
    min_count = counts.min()
    balanced_segs, balanced_labels = [], []
    for cls in unique_classes:
        idx = np.where(labels == cls)[0]
        chosen = np.random.choice(idx, min_count, replace=False)
        balanced_segs.append(segments[chosen])
        balanced_labels.append(labels[chosen])
    return np.concatenate(balanced_segs), np.concatenate(balanced_labels)


def load_and_label_data(data_dir, window_size=None, stride=10, include_subjects=None):
    """
    Loads CSV files from data_dir, assigns labels based on filename keywords,
    and segments each file with a sliding window.

    Files follow the naming convention: S{n}_t{n}_{activity}[_{anomaly}].csv
    where Sn is the subject and tn is the trial. The label is determined by the
    first matching keyword in LABEL_MAPPING (priority order: falldown > shaking > ...).

    Args:
        data_dir (str): Directory containing CSV files.
        window_size (int): Rows per segment. If None, uses the minimum file length.
        stride (int): Number of time steps to advance the sliding window.
        include_subjects (list, optional): If provided (e.g. ['S1','S3']), only load
            files belonging to those subjects. Used for LOSO cross-validation.

    Returns:
        data_list (list): List of numpy arrays, each of shape (window_size, n_features).
        labels (list): Integer label for each segment.
    """
    data_list = []
    labels = []
    valid_files = []

    # First pass: read files, filter by subject and label
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

    if not valid_files:
        raise ValueError(f"No valid files loaded from {data_dir} with subjects={include_subjects}")

    # If window_size is not provided, use the minimum number of rows across files
    if window_size is None:
        window_size = min(sd.shape[0] for sd, _ in valid_files)
        print(f"No window_size specified. Using minimum file length: {window_size}")

    # Slide a window over each file
    for sensor_data, label in valid_files:
        L = sensor_data.shape[0]
        if L < window_size:
            print(f"File length {L} < window_size {window_size}. Skipping.")
            continue
        for i in range(0, L - window_size + 1, stride):
            data_list.append(sensor_data[i:i + window_size])
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

def train_val_split(segments, labels, val_size=0.2, random_state=42):
    """
    Split into train and validation sets (stratified by label).
    Used within each LOSO fold to reserve 20% of training data for early stopping.

    Returns: X_train, X_val, y_train, y_val
    """
    return train_test_split(
        segments, labels, test_size=val_size, random_state=random_state, stratify=labels
    )


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
