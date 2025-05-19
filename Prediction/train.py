import torch
import torch.nn as nn
import torch.optim as optim
import copy

def train_model(model, train_loader, val_loader, config, device):
    """
    Train the model with early stopping. Returns:
      - best_model: the model with the best validation performance
      - train_loss_history: list of training losses per epoch
      - val_loss_history: list of validation losses per epoch
    """
    epochs = config['epochs']
    learning_rate = config.get('learning_rate', 0.001)
    patience = config.get('early_stopping_patience', 5)
    if(config['criterion'] == "mae"): criterion = nn.SmoothL1Loss()
    else: criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_train_loss)
        
        # Evaluate on validation set
        epoch_val_loss = eval_model(model, val_loader, criterion, device)
        val_loss_history.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")
        
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model = copy.deepcopy(model)
            print("  New best model found!")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  Early stopping triggered.")
                break
                
    return best_model, train_loss_history, val_loss_history

def eval_model(model, data_loader, criterion, device):
    """
    Evaluate the model on the provided data_loader.
    Returns the average loss over the dataset.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_X.size(0)
    avg_loss = total_loss / len(data_loader.dataset)
    print(f"Evaluation Loss: {avg_loss:.4f}")
    return avg_loss
