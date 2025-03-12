import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)

# Data loaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Define CNN model
class Cifar_CNN(nn.Module):
    def __init__(self):
        super(Cifar_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # First convolutional layer
        self.conv2 = nn.Conv2d(6, 16, 5)  # Second convolutional layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # First fully connected layer
        self.fc2 = nn.Linear(120, 84)  # Second fully connected layer
        self.fc3 = nn.Linear(84, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply first convolution + ReLU
        x = F.max_pool2d(x, 2, 2)  # Apply max pooling
        x = F.relu(self.conv2(x))  # Apply second convolution + ReLU
        x = F.max_pool2d(x, 2, 2)  # Apply max pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers
        x = F.relu(self.fc1(x))  # First fully connected layer + ReLU
        x = F.relu(self.fc2(x))  # Second fully connected layer + ReLU
        x = self.fc3(x)  # Output layer
        return x

# Instantiate model, loss function, and optimizer
model = Cifar_CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
for epoch in range(10):
    model.train()
    loss_epoch = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()  # Zero the parameter gradients
        predictions = model(images)  # Forward pass
        loss = criterion(predictions, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize the model

        _, predicted = torch.max(predictions.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss_epoch += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {loss_epoch:.4f}, Training Accuracy: {(correct / total) * 100:.2f}%")

# Evaluation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        predictions = model(images)
        _, predicted = torch.max(predictions.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {(correct / total) * 100:.2f}%")
