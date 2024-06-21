import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import cv2 as cv


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(32, 128, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.50)
        self.dropout3 = nn.Dropout(0.75)
        self.ln1 = nn.Linear(61952, 128)   # 22x22x128 , 9216/128 = 61952/x
        self.ln2 = nn.Linear(1024, 128)
        self.ln3 = nn.Linear(128, 10)

    def forward(self, x, y=None):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.ln2(x)
        x = F.relu(x)  
        x = self.dropout3(x)
        x = self.ln3(x)
        if y != None:
            loss = F.cross_entropy(x, y)
        logits = F.softmax(x, dim=1)
        return logits, loss
        
        