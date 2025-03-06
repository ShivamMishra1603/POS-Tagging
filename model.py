import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

def trainLogReg(train_data, dev_data, learning_rate, l2_penalty, num_epochs=100):
    X_train, y_train = train_data
    X_dev, y_dev = dev_data

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_dev_tensor = torch.tensor(X_dev, dtype=torch.float32)
    y_dev_tensor = torch.tensor(y_dev, dtype=torch.long)

    input_size = X_train.shape[1]
    output_size = len(np.unique(y_train))
    
    model = LogisticRegressionModel(input_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_penalty)
    
    train_losses = []
    train_accuracies = []
    dev_losses = []
    dev_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        train_loss = criterion(outputs, y_train_tensor)
        
        train_loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        train_accuracy = (predicted == y_train_tensor).float().mean().item()
        
        model.eval()
        with torch.no_grad():
            dev_outputs = model(X_dev_tensor)
            dev_loss = criterion(dev_outputs, y_dev_tensor)
            _, dev_predicted = torch.max(dev_outputs, 1)
            dev_accuracy = (dev_predicted == y_dev_tensor).float().mean().item()

        train_losses.append(train_loss.item())
        train_accuracies.append(train_accuracy)
        dev_losses.append(dev_loss.item())
        dev_accuracies.append(dev_accuracy)

    return model, train_losses, train_accuracies, dev_losses, dev_accuracies