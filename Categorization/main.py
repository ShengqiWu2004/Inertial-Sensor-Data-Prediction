import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from data_loader import (
    load_and_label_data,
    normalize_segments,
    train_val_test_split_data,
    SensorDataset,
    LABEL_MAPPING
)
from train import train_model, eval_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train & Evaluate Sensor Model with PyTorch")
    parser.add_argument("--config", type=str, required=True,
                        help="Configuration file name (without extension) in the 'config/' directory")
    return parser.parse_args()

def load_config(name):
    config_path = os.path.join("config", f"{name}.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_all_data_augmented(data_dir, window_size, stride):
    # Load and augment the data using sliding window
    data_list, labels = load_and_label_data(data_dir, window_size=window_size, stride=stride)
    segments = np.array(data_list)
    segments, scaler = normalize_segments(segments)
    return segments, np.array(labels), scaler

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Use augmented data (set window_size and stride in config)
    data_dir = "./dataset/train"  # adjust as needed
    window_size = config.get("window_size", 100)  # e.g., 100 timesteps per segment
    stride = config.get("stride", 10)              # slide by 10 time steps
    
    segments, labels, scaler = load_all_data_augmented(data_dir, window_size, stride)
    print("Augmented data shape:", segments.shape)
    
    # Split dataset: e.g., 60% train, 20% val, 20% test
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_data(segments, labels, test_size=0.2, val_size=0.2)
    
    # Create PyTorch datasets
    num_classes = len(LABEL_MAPPING)
    train_dataset = SensorDataset(X_train, y_train, one_hot=False)
    val_dataset = SensorDataset(X_val, y_val, one_hot=False)
    test_dataset = SensorDataset(X_test, y_test, one_hot=False)
    
    # Set device and DataLoaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config.get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Train the model (for example, using the hybrid model if configured)
    model, train_loss_history, val_loss_history, criterion, device = train_model(
        train_loader, val_loader, config, device, X_train
    )
    
    # Save best model
    output_model_path = config.get("model_save_path", "./output_model/best_model.pth")
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(model.state_dict(), output_model_path)
    print(f"Best model saved to {output_model_path}")
    
    # Evaluate on test set
    eval_model(model, test_loader, criterion, device)

if __name__ == '__main__':
    import numpy as np  # ensure numpy is imported for np.array in load_all_data_augmented
    main()
