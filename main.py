from src.data_loader import load_mnist_data, create_data_loaders
from src.train import trainModel
from src.visualize import plot_metrics
import torch

def main():
    # File paths
    train_data_path = 'datasets/train-images-idx3-ubyte/train-images-idx3-ubyte'
    train_labels_path = 'datasets/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    test_data_path = 'datasets/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
    test_labels_path = 'datasets/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

    # Load data
    train_data, train_labels, test_data, test_labels = load_mnist_data(train_data_path, train_labels_path, test_data_path, test_labels_path)
    
    # Create DataLoaders
    train_loader, test_loader = create_data_loaders(train_data, train_labels, test_data, test_labels)

    # Train the model
    trainAcc, testAcc, losses, net = trainModel('Adam', 0.001, epochs=100, train_loader=train_loader, test_loader=test_loader)

    # Save the trained model
    model_save_path = 'models/mnist_model.pth'
    torch.save(net.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot and save the training metrics
    plot_save_path = 'models/training_metrics.png'
    plot_metrics(losses, trainAcc, testAcc, save_path=plot_save_path)
    print(f"Training metrics plot saved to {plot_save_path}")

if __name__ == "__main__":
    main()
