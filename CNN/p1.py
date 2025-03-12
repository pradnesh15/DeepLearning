import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.optim as optim
import torch.nn as nn

# Define transformations for the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 training and testing datasets
train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)

# Define data loaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Define the CNN model
class Cifar_CNN(nn.Module):  # Corrected inheritance from nn.Module
    def __init__(self):
        super(Cifar_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # (input_channels, output_channels, kernel_size)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Corrected number of input features for fc1
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # Output layer with 10 classes (CIFAR-10)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # Apply first convolution and ReLU
        x = F.max_pool2d(x, 2, 2)       # Apply first max pooling
        x = F.relu(self.conv2(x))       # Apply second convolution and ReLU
        x = F.max_pool2d(x, 2, 2)       # Apply second max pooling
        x = x.view(-1, 16 * 5 * 5)      # Flatten the tensor for fully connected layers
        x = F.relu(self.fc1(x))         # First fully connected layer with ReLU
        x = F.relu(self.fc2(x))         # Second fully connected layer with ReLU
        x = self.fc3(x)                 # Output layer
        return x

# Instantiate the model, loss function, and optimizer
model = Cifar_CNN().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # Use GPU if available
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
for epoch in range(10):  # Reduced epochs to 10 for faster training
    model.train()
    loss_epoch = 0
    for images, labels in train_loader:
        images, labels = images.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), labels.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Forward pass
        predictions = model(images)
        loss = criterion(predictions, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

    print(f"Epoch [{epoch+1}/10], Loss: {loss_epoch/len(train_loader):.4f}")

# Evaluation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), labels.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        predictions = model(images)
        _, predicted = torch.max(predictions.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy: {(correct / total) * 100:.2f}%")
