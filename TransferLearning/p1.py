import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models


t=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_dataset=CIFAR10(root='./data',train=True,download=True,transform=t)
test_dataset=CIFAR10(root='./data',train=False,download=True,transform=t)
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)


model=models.resnet18(pretrained=True)
for param in model.parameters():
  param.requires_grad=False
in_features=model.fc.in_features
model.fc=nn.Linear(in_features,10)
criterian=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)



for epochs in range(5):
  model.train()
  running_loss=0
  for data,target in train_loader:
    optimizer.zero_grad()
    output=model(data)
    loss=criterian(output,target)
    loss.backward()
    optimizer.step()
    running_loss+=loss.item()
  print(f"Epoch {epochs+1}, Loss: {running_loss/len(train_loader)}")



model.eval()
with torch.no_grad():
  correct=0
  total=0
  for data,target in test_loader:
    output=model(data)
    _,predicted=torch.max(output.data,1)
    total+=target.size(0)
    correct+=(predicted==target).sum().item()
  print(f"Accuracy: {(correct/total)*100}%")
