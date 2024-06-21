import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import cv2 as cv
import argparse
from icecream import ic
import numpy as np

device = 'cpu'
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"


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
        

def get_dataset(B):
    dataset_train = datasets.MNIST("./data", train=True, transform=transforms.ToTensor(), download=True)
    dataset_test = datasets.MNIST("./data", train=False, transform=transforms.ToTensor())
    
    train_loader = torch.utils.data.DataLoader(dataset_train, B, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, B, shuffle=True)
    
    return train_loader, test_loader


def train(model, device, train_loader, optimizer, num_epoch):
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        
def val(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            pred = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            print("Loss: {:.4f} | Pred: {pred}").format(loss.item(), pred)


def infer(model, device, image):
    model.to(device)
    results = model(image)
    return results


def preprocess(image_path: str):
    image = cv.imread(image_path, 0)
    image = cv.bitwise_not(image)
    image = cv.copyMakeBorder(image, 2, 2, 2, 2, cv.BORDER_CONSTANT, value=0)
    image = cv.resize(image, (28, 28))
    # image.imshow()
    image = image.astype(np.float32)
    image = image / 255
    ic(image.shape)
    image = np.expand_dims(image, axis=0)
    ic(image.shape)
    image = np.expand_dims(image, axis=0)
    ic(image.shape)   # batch, channel, height, width
    image = torch.from_numpy(image)
    return image

def postprocess(results):
    results = torch.Tensor.detach(results)
    results = torch.Tensor.numpy(results)
    return np.argmax(results)


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument("-b", "--num_batch", type=int, help="Number of batches")
    parser.add_argument("-p", "--image_path", type=str, help="Path to test image")
    parser.add_argument("-v", '--verbose', action="store_true", default=False, help='Prints everything')

    args = parser.parse_args()

    if args.verbose:
        ic.enable()
    else:
        ic.disable()
        
    model = Net()
    model = model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1e-3)
    
    
    
    if args.image_path != None:
        image = preprocess(args.image_path)
        result = infer(model, device, image)
        print(f"Result for the image: {result}")