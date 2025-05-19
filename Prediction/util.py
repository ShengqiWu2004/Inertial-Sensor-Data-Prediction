import matplotlib.pyplot as plt
import os

def plot_losses(train_losses, val_losses, save_path="figures/loss_curve.png"):
    """
    Plots the training and validation loss curves.
    
    Args:
        train_losses (list or array): Training loss values per epoch.
        val_losses (list or array): Validation loss values per epoch.
        save_path (str): Path where the plot image will be saved.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    
    # Ensure the directory exists
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curves saved to {save_path}")
