import torch
import numpy as np
from src.model import mnistNet

def mnistModel(o='SGD', lr=0.01):
    # Create model instance
    net = mnistNet()

    # Loss function
    lossfun = torch.nn.CrossEntropyLoss()

    # Optimizer
    optifun = getattr(torch.optim, o)
    optimizer = optifun(net.parameters(), lr=lr)

    return net, lossfun, optimizer

def trainModel(o, lr, epochs=100, train_loader=None, test_loader=None):
    if train_loader is None or test_loader is None:
        raise ValueError("train_loader and test_loader cannot be None")

    # Create a new model
    net, lossfun, optimizer = mnistModel(o, lr)

    # Initialize losses
    losses = torch.zeros(epochs)
    trainAcc = []
    testAcc = []

    # Looping over epochs
    for epochi in range(epochs):
        batchAcc = []
        batchLoss = []
        
        for X, y in train_loader:
            # Forward pass and loss
            yHat = net(X)
            loss = lossfun(yHat, y)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Loss from this batch
            batchLoss.append(loss.item())

            # Compute accuracy
            matches = torch.argmax(yHat, axis=1) == y
            matchesNumeric = matches.float()
            accuracyPct = 100 * torch.mean(matchesNumeric)
            batchAcc.append(accuracyPct)

        # Get average training accuracy
        trainAcc.append(np.mean(batchAcc))
        
        # Get average losses across the batches
        losses[epochi] = np.mean(batchLoss)
        
        # Test accuracy
        X, y = next(iter(test_loader))
        with torch.no_grad():
            yHat = net(X)
        testAcc.append(100 * torch.mean((torch.argmax(yHat, axis=1) == y).float()))

    return trainAcc, testAcc, losses, net
