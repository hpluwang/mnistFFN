import idx2numpy
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_mnist_data(train_data_path, train_labels_path, test_data_path, test_labels_path):
    # Load train and test data
    train_images = idx2numpy.convert_from_file(train_data_path)
    train_labels = idx2numpy.convert_from_file(train_labels_path)
    test_images = idx2numpy.convert_from_file(test_data_path)
    test_labels = idx2numpy.convert_from_file(test_labels_path)

    # Flatten images
    train_images_flat = train_images.reshape(train_images.shape[0], -1)
    test_images_flat = test_images.reshape(test_images.shape[0], -1)

    # Convert numpy arrays to PyTorch tensors
    train_data = torch.tensor(train_images_flat, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(test_images_flat, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    return train_data, train_labels, test_data, test_labels

def create_data_loaders(train_data, train_labels, test_data, test_labels, batch_size=128):
    # Convert PyTorch tensors to datasets
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)  # Using full batch for test

    return train_loader, test_loader
