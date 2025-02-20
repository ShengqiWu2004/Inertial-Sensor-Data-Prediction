import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from data_loader import load_and_segment_data, SensorDataset, train_val_test_split
from models.cnn_model import CNNModel
from models.lstm import LSTMModel
from train import train_model, eval_model
from util import plot_losses

def parse_args():
    parser = argparse.ArgumentParser(description="Train & Evaluate Sensor Forecasting Model")
    parser.add_argument("--config", type=str, required=True,
                        help="Configuration file name (without extension) in the 'config/' directory")
    return parser.parse_args()

def load_config(name):
    config_path = os.path.join("config", f"{name}.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Load and segment data (with balancing)
    X, y, labels = load_and_segment_data(
        data_dir="dataset/train",
        window_size=config['window_size'],
        predict_size=config['predict_size'],
        balance_config=config
    )
    
    # Split data into train, validation, and test sets.
    X_train, X_val, X_test, y_train, y_val, y_test,_,_,_ = train_val_test_split(
        X, y, labels, test_size=0.2, val_size=0.4, random_state=42
    )
    
    # Create datasets and dataloaders (normalization is done in SensorDataset)
    train_dataset = SensorDataset(X_train, y_train, normalize=True)
    val_dataset   = SensorDataset(X_val, y_val, normalize=True, scaler=train_dataset.scaler)
    test_dataset  = SensorDataset(X_test, y_test, normalize=True, scaler=train_dataset.scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create the CNN model for regression
    if config["model_type"] == "cnn":  
        model = CNNModel(window_size=config['window_size'],
                     predict_size=config['predict_size'],
                     num_features=6)
    elif config["model_type"] == "lstm":
        model = LSTMModel(input_size=6,hidden_size=config.get('hidden_size', 64), num_layers=config.get('num_layers', 2),predict_size=config['predict_size'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Train the model (early stopping is handled in train_model)
    best_model, train_losses, val_losses = train_model(model, train_loader, val_loader, config, device)
    
    # Save the best model
    output_model_path = config.get('model_save_path', "./output_model/best_model.pth")
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(best_model.state_dict(), output_model_path)
    print(f"Best model saved to {output_model_path}")
    
    # Evaluate on test set using the separate evaluation function
    criterion = torch.nn.MSELoss()  # Using MSE as the regression loss criterion
    print("\nEvaluating on test set:")
    test_loss = eval_model(best_model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Plot and save loss curves
    plot_losses(train_losses, val_losses, save_path="figures/loss_curve.png")

if __name__ == '__main__':
    main()
