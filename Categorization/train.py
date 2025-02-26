import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Import models from your modules
from models.cnn_model import CNNModel
from models.lstm_model import LSTMModel
from models.hybrid_model import HybridCNNSoftmaxModel, HybridCNNSoftmaxAttentionModel
from models.hybrid_moe_model import HybridMixtureOfExpertsModel
from data_loader import train_val_test_split_data, SensorDataset, LABEL_MAPPING

def train_model(train_loader, val_loader, config, device, X_train):
    num_classes = config.get('num_classes', 6)
    model_type = config.get('model_type', 'cnn').lower()
    input_features = X_train.shape[-1]
    
    # Select model based on configuration
    if model_type == 'lstm':
        model = LSTMModel(
            num_classes=num_classes,
            input_features=input_features,
            hidden_size=config.get('lstm_hidden_size', 128),
            num_layers=config.get('lstm_num_layers', 2),
            bidirectional=config.get('lstm_bidirectional', False),
            dropout=config.get('dropout', 0.5),
            weight_threshold=config.get('lstm_weight_threshold', 60),
            max_weight=config.get('lstm_max_weight', 1.2)
        ).to(device)
    elif model_type == 'hybrid_moe':
        full_length = X_train.shape[1]
        model = HybridMixtureOfExpertsModel(
            num_classes=num_classes,
            window_size=full_length,
            input_features=input_features,
            num_experts=config.get('num_experts', 3),
            dropout=config.get('dropout', 0.5)
        ).to(device)
    elif model_type == 'hybrid_attention':
        full_length = X_train.shape[1]
        model = HybridCNNSoftmaxAttentionModel(
            num_classes=num_classes,
            window_size=full_length,
            input_features=input_features
        ).to(device)
    elif model_type == 'hybrid':
        full_length = X_train.shape[1]
        model = HybridCNNSoftmaxModel(
            num_classes=num_classes,
            window_size=full_length,
            input_features=input_features
        ).to(device)
    else:
        full_length = X_train.shape[1]
        model = CNNModel(
            num_classes=num_classes,
            window_size=full_length,
            input_features=input_features
        ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    num_epochs = config.get('epochs', 10)
    
    patience = config.get('early_stopping_patience', 5)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if labels.ndim > 1:
                labels = torch.argmax(labels, dim=1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if labels.ndim > 1:
                    labels = torch.argmax(labels, dim=1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_loss_history.append(val_epoch_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")
        
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs. No improvement for {patience} consecutive epochs.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model, train_loss_history, val_loss_history, criterion, device

def eval_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if labels.ndim > 1:
                labels = torch.argmax(labels, dim=1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = test_loss / total
    accuracy = correct / total
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
