import argparse
import os
import json
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from data_loader import (
    load_and_label_data,
    normalize_segments,
    balance_segments,
    train_val_split,
    get_all_subjects,
    SensorDataset,
    LABEL_MAPPING,
)
from train import train_model, eval_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train & Evaluate Sensor Categorization Model (LOSO CV)")
    parser.add_argument("--config", type=str, required=True,
                        help="Configuration file name (without extension) in the 'config/' directory")
    return parser.parse_args()


def load_config(name):
    config_path = os.path.join("config", f"{name}.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_subject_data(data_dir, window_size, stride, include_subjects):
    data_list, labels = load_and_label_data(
        data_dir, window_size=window_size, stride=stride,
        include_subjects=include_subjects
    )
    return np.array(data_list), np.array(labels)


def main():
    args = parse_args()
    config = load_config(args.config)
    cfg_name = args.config

    data_dir = "../dataset/train"
    window_size = config.get("window_size", 100)
    stride = config.get("stride", 10)
    batch_size = config.get("batch_size", 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output directories (separate from legacy outputs)
    loso_model_dir  = os.path.join("output_model", f"loso_{cfg_name}")
    loso_result_dir = "loso_results"
    for d in (loso_model_dir, loso_result_dir):
        os.makedirs(d, exist_ok=True)

    subjects = get_all_subjects(data_dir)
    print(f"Found {len(subjects)} subjects for LOSO cross-validation: {subjects}")

    fold_results = {}

    for test_subject in subjects:
        train_subjects = [s for s in subjects if s != test_subject]
        print(f"\n{'='*60}")
        print(f"LOSO fold | test: {test_subject}  train: {train_subjects}")
        print(f"{'='*60}")

        # --- Load + balance training data from the 4 remaining subjects ---
        segs_all, labels_all = load_subject_data(
            data_dir, window_size, stride, include_subjects=train_subjects
        )
        # Under-sample each class to the smallest class count (training only)
        segs_all, labels_all = balance_segments(segs_all, labels_all)

        # Fit scaler on training subjects only
        segs_all, scaler = normalize_segments(segs_all)

        # 80 / 20 train / val split — val used only for early stopping (patience=10)
        X_train, X_val, y_train, y_val = train_val_split(segs_all, labels_all, val_size=0.2)

        # --- All data from the held-out subject is the test set ---
        X_test, y_test = load_subject_data(
            data_dir, window_size, stride, include_subjects=[test_subject]
        )
        X_test, _ = normalize_segments(X_test, scaler=scaler)

        train_dataset = SensorDataset(X_train, y_train, one_hot=False)
        val_dataset   = SensorDataset(X_val,   y_val,   one_hot=False)
        test_dataset  = SensorDataset(X_test,  y_test,  one_hot=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

        print(f"  Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")

        model, train_loss_history, val_loss_history, criterion, device = train_model(
            train_loader, val_loader, config, device, X_train
        )

        # Accuracy on held-out subject
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, predicted = torch.max(model(inputs), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        fold_acc = correct / total
        print(f"  {test_subject} accuracy: {fold_acc*100:.2f}%")

        fold_model_path = os.path.join(loso_model_dir, f"{test_subject}.pth")
        torch.save(model.state_dict(), fold_model_path)

        fold_results[test_subject] = {
            "accuracy":      round(fold_acc, 4),
            "train_samples": len(train_dataset),
            "val_samples":   len(val_dataset),
            "test_samples":  len(test_dataset),
            "epochs_trained": len(train_loss_history),
            "final_val_loss": round(float(val_loss_history[-1]), 6),
        }

    # --- LOSO summary ---
    accs = [fold_results[s]["accuracy"] for s in subjects]
    summary = {
        "config":        cfg_name,
        "model":         config["model_type"],
        "folds":         fold_results,
        "mean_accuracy": round(float(np.mean(accs)), 4),
        "std_accuracy":  round(float(np.std(accs)),  4),
    }

    result_path = os.path.join(loso_result_dir, f"{cfg_name}_loso_results.json")
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("Leave-One-Subject-Out Cross-Validation Results:")
    for subj in subjects:
        print(f"  {subj}: Accuracy = {fold_results[subj]['accuracy']*100:.2f}%")
    print(f"  Mean Accuracy: {summary['mean_accuracy']*100:.2f}%")
    print(f"  Std  Accuracy: {summary['std_accuracy']*100:.2f}%")
    print(f"  Results saved to {result_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
