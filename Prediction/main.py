import argparse
import os
import json
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from data_loader import (
    load_and_segment_data,
    train_val_split,
    get_all_subjects,
    SensorDataset,
)
from models.cnn_model import CNNModel
from models.lstm import LSTMModel
from models.rnn import VanillaRNNModel
from models.ntflstm import LSTMAutoregressive
from train import train_model, eval_model
from util import plot_losses


def parse_args():
    parser = argparse.ArgumentParser(description="Train & Evaluate Sensor Forecasting Model (LOSO CV)")
    parser.add_argument("--config", type=str, required=True,
                        help="Configuration file name (without extension) in the 'config/' directory")
    return parser.parse_args()


def load_config(name):
    config_path = os.path.join("config", f"{name}.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(config, device):
    model_type = config["model_type"]
    if model_type == "cnn":
        model = CNNModel(window_size=config['window_size'],
                         predict_size=config['predict_size'],
                         num_features=6)
    elif model_type == "lstm":
        model = LSTMModel(input_size=6,
                          hidden_size=config.get('hidden_size', 64),
                          num_layers=config.get('num_layers', 2),
                          predict_size=config['predict_size'])
    elif model_type == "ntflstm":
        model = LSTMAutoregressive(input_size=6,
                                   hidden_size=config.get('hidden_size', 64),
                                   num_layers=config.get('num_layers', 2),
                                   predict_size=config['predict_size'])
    elif model_type == "rnn":
        model = VanillaRNNModel(input_size=6,
                                hidden_size=config.get('hidden_size', 64),
                                num_layers=config.get('num_layers', 2),
                                predict_size=config['predict_size'])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model.to(device)


def main():
    args = parse_args()
    config = load_config(args.config)
    cfg_name = args.config  # used for naming output files

    data_dir = "dataset/train"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output directories (separate from legacy single-run outputs)
    loso_model_dir = os.path.join("output_model", f"loso_{cfg_name}")
    loso_fig_dir   = os.path.join("figures",      f"loso_{cfg_name}")
    loso_result_dir = "loso_results"
    for d in (loso_model_dir, loso_fig_dir, loso_result_dir):
        os.makedirs(d, exist_ok=True)

    # Discover all subjects (S1–S5) from filenames (S{n}_t{n}_*.csv)
    subjects = get_all_subjects(data_dir)
    print(f"Found {len(subjects)} subjects for LOSO cross-validation: {subjects}")

    fold_results = {}

    for test_subject in subjects:
        train_subjects = [s for s in subjects if s != test_subject]
        print(f"\n{'='*60}")
        print(f"LOSO fold | test: {test_subject}  train: {train_subjects}")
        print(f"{'='*60}")

        # --- Load training data from the 4 remaining subjects ---
        # balance=False: this is a regression task (MSE), not classification.
        # Under-sampling to the rarest class would discard ~75% of training data
        # without any benefit for a regression objective.
        X_all, y_all, labels_all = load_and_segment_data(
            data_dir=data_dir,
            window_size=config['window_size'],
            predict_size=config['predict_size'],
            balance_config=config,
            include_subjects=train_subjects,
            balance=False,
        )

        # Split 80 / 20 within the training subjects.
        # The 20% validation split is used exclusively for early stopping
        # (patience = 10 epochs), not for model selection or hyperparameter tuning.
        X_train, X_val, y_train, y_val, _, _ = train_val_split(
            X_all, y_all, labels_all, val_size=0.2
        )

        # --- Load ALL data from the held-out subject as the test set ---
        X_test, y_test, _ = load_and_segment_data(
            data_dir=data_dir,
            window_size=config['window_size'],
            predict_size=config['predict_size'],
            balance_config=config,
            include_subjects=[test_subject],
            balance=False,
        )

        # Fit scaler on training data; apply to validation and test
        train_dataset = SensorDataset(X_train, y_train, normalize=True)
        val_dataset   = SensorDataset(X_val,   y_val,   normalize=True, scaler=train_dataset.scaler)
        test_dataset  = SensorDataset(X_test,  y_test,  normalize=True, scaler=train_dataset.scaler)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'], shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=config['batch_size'], shuffle=False)

        print(f"  Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")

        # Fresh model for each fold
        model = build_model(config, device)

        # Train with early stopping monitored on the validation set
        best_model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, config, device
        )

        # Evaluate on the held-out subject
        criterion = torch.nn.MSELoss()
        test_loss = eval_model(best_model, test_loader, criterion, device)
        print(f"  {test_subject} test MSE: {test_loss:.6f}")

        # Save this fold's model and loss curve (unique filenames, no overwrite of legacy outputs)
        fold_model_path = os.path.join(loso_model_dir, f"{test_subject}.pth")
        torch.save(best_model.state_dict(), fold_model_path)

        fold_fig_path = os.path.join(loso_fig_dir, f"{test_subject}_loss.png")
        plot_losses(train_losses, val_losses, save_path=fold_fig_path)

        fold_results[test_subject] = {
            "test_mse":      round(float(test_loss), 6),
            "train_samples": len(train_dataset),
            "val_samples":   len(val_dataset),
            "test_samples":  len(test_dataset),
            "epochs_trained": len(train_losses),
            "final_val_loss": round(float(val_losses[-1]), 6),
        }

    # --- LOSO summary ---
    losses = [fold_results[s]["test_mse"] for s in subjects]
    summary = {
        "config":   cfg_name,
        "model":    config["model_type"],
        "folds":    fold_results,
        "mean_mse": round(float(np.mean(losses)), 6),
        "std_mse":  round(float(np.std(losses)),  6),
    }

    result_path = os.path.join(loso_result_dir, f"{cfg_name}_loso_results.json")
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("Leave-One-Subject-Out Cross-Validation Results:")
    for subj in subjects:
        print(f"  {subj}: MSE = {fold_results[subj]['test_mse']:.6f}")
    print(f"  Mean MSE : {summary['mean_mse']:.6f}")
    print(f"  Std  MSE : {summary['std_mse']:.6f}")
    print(f"  Results saved to {result_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
