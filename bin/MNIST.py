import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import cv2 as cv


class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.cv1 = nn.Conv2d(1, 32, 3, 1)
        self.cv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.50)
        self.ln1 = nn.Linear(9216, 128)
        self.ln1 = nn.Linear(128, 10)