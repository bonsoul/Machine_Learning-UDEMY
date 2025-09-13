import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#define transformation
transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5))
])


#load the datsets
train_dataset = datasets.MNIST(root="./data", train=True, transform=transforms, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transforms, download=True)


#create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
