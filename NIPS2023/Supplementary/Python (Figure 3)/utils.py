import os
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets 
import numpy as np
import matplotlib.pyplot as plt


def CIFAR10_dataset():
    if not os.path.isdir("data"):
        os.mkdir("data")
    # Download CIFAR10 dataset and set the valset as the test test
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_set = datasets.CIFAR10('data/CIFAR10', download=True, train=False, transform=transform)
    train_set = datasets.CIFAR10("data/CIFAR10", download=True, train=True, transform=transform)
    return train_set, test_set
    nn.Conv2d



class CIFAR10_ConvNet(nn.Module):
    def __init__(self):
        super(CIFAR10_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def accuracy(yhat, labels):
    _, indices = yhat.max(1)
    return (indices == labels).sum().data.item() / float(len(labels))

class AverageCalculator():
    def __init__(self):
        self.reset() 
    
    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
    
    def update(self, val, n=1):
        assert(n > 0)
        self.sum += val * n 
        self.count += n
        self.avg = self.sum / float(self.count)

def plot_train_stats(train_loss, val_loss, train_acc, val_acc, directory, acc_low=0):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,6), sharey='row')
    axes[0][0].plot(np.array(train_loss))
    axes[0][0].set_title("Train loss")
    axes[0][1].plot(np.array(val_loss))
    axes[0][1].set_title("Val loss")
    axes[1][0].plot(np.array(train_acc))
    axes[1][0].set_title("Train Accuracy")
    axes[1][0].set_ylim(acc_low, 1)
    axes[1][1].plot(np.array(val_acc))
    axes[1][1].set_title("Val Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'train_stats.pdf'))
    plt.savefig(os.path.join(directory, 'train_stats.png'))
    plt.close()
