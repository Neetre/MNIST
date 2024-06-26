'''
Definition of the NN and setting up
some functions for the GUI.

Neetre 2024
'''

import argparse
from icecream import ic

import cv2 as cv
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


device = 'cpu'
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"


class Net_load(nn.Module):
    """
    The Neural Network for the MNIST dataset.

    Args:
        nn (class nn): The class nn from torch
    """

    def __init__(self):
        """
        Setting up the layers of the NN.
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.conv3 = nn.Conv2d(64, 128, 3, 1)
        # self.conv4 = nn.Conv2d(128, 512, 3, 1)
        # self.conv5 = nn.Conv2d(512, 1024, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.50)
        # self.dropout3 = nn.Dropout(0.75)
        self.fc1 = nn.Linear(9216, 128)  # 7*7*1024, 1024
        self.fc2 = nn.Linear(128, 10)
        # self.ln3 = nn.Linear(128, 10)

    def forward(self, x) -> torch.tensor:
        """
        Forward pass of the NN.

        Args:
            x (torch.tensor): The input tensor

        Returns:
            torch.tensor: The output tensor
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # x = self.conv3(x)
        # x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.conv4(x)
        # x = F.relu(x)
        # x = self.conv5(x)
        # x = F.relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # x = F.relu(x)
        # x = self.dropout3(x)
        # x = self.ln3(x)
        x = F.log_softmax(x, dim=1)
        return x
    

class Net(nn.Module):
    """
    The Neural Network for the MNIST dataset.

    Args:
        nn (class nn): The class nn from torch
    """

    def __init__(self):
        """
        Setting up the layers of the NN.
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 512, 3, 1)
        self.conv5 = nn.Conv2d(512, 1024, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.50)
        self.dropout3 = nn.Dropout(0.75)
        self.fc1 = nn.Linear(9216, 128)  # 7*7*1024, 1024
        self.fc2 = nn.Linear(128, 10)
        self.ln3 = nn.Linear(128, 10)

    def forward(self, x) -> torch.tensor:
        """
        Forward pass of the NN.

        Args:
            x (torch.tensor): The input tensor

        Returns:
            torch.tensor: The output tensor
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.ln3(x)
        x = F.log_softmax(x, dim=1)
        return x


def get_dataset(B):
    """
    Get the MNIST dataset, and divede it in batches.

    Args:
        B (int): The number of batches

    Returns:
        _type_: The train and test loader
    """
    dataset_train = datasets.MNIST("../data", train=True,
                                   transform=transforms.ToTensor(), download=True)
    dataset_test = datasets.MNIST("../data", train=False,
                                  transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset_train, B, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, B, shuffle=True)

    return train_loader, test_loader


def train(model, device, train_loader, optimizer, epoch):
    """
    Train the NN.

    Args:
        model (Net): The NN to train
        device (str): The device to use (cpu, cuda, mps)
        train_loader (): the loader of the train dataset
        optimizer (optim): The optimizer to use (Adadelta, could also be AdamW)
        epoch (int): The number of epochs
    """
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)  # feedforward
        loss = F.cross_entropy(logits, y)
        loss.backward()  # backpropagation
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def val(model, device, test_loader):
    """
    Validate the NN.

    Args:
        model (Net): The NN to validate
        device (str): The device to use (cpu, cuda, mps)
        test_loader (): The loader of the test dataset
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            test_loss += F.cross_entropy(logits, y)
            pred = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f"Val Loss: {test_loss:.4f}  |  Accuracy: {correct}/{len(test_loader.dataset)}")


def set_ic_gradio(on: bool):
    """
    Set the icecream module for the GUI.

    Args:
        on (bool): Enable or disable the module
    """
    if on:
        ic.enable()
    else:
        ic.disable()


def load_model(model: Net, compile_: bool):
    """
    Load a pre-trained model.

    Args:
        model (Net): The model to load
        compile (bool): Compile the model

    Returns:
        Net: The loaded model
    """
    try:
        model.load_state_dict(torch.load("./model/mnist_cnn.pt", map_location=device))
    except FileNotFoundError:
        print("Couldn't find the pre-trained model.")
        print("Try training one, or check the path.")
    except Exception as e:
        print(f"Error: {e}")
    model = model.to(device)
    if compile_:
        model = torch.compile(model)
    model.eval()
    return model


def infer(model, device, image):
    """
    Infer the image.

    Args:
        model (Net): The model to use
        device (str): The device to use (cpu, cuda, mps)
        image (torch.tensor): The image to infer

    Returns:
        torch.tensor: The result of the inference
    """
    model.to(device)
    results = model(image)
    return results


def preprocess(image_path: str) -> torch.tensor:
    """
    Preprocess the image.

    Args:
        image_path (str): The path to the image

    Returns:
        Tensor: The preprocessed image
    """
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


def get_device():
    """
    Get the device to use.

    Returns:
        str: The device to use
    """
    return device


def postprocess(results):
    """
    Postprocess the results of the infer.

    Args:
        results (torch.tensor): The results of the infer

    Returns:
        int: The result of the infer (the number predicted)
    """
    results = torch.Tensor.detach(results)
    results = torch.Tensor.numpy(results)
    return np.argmax(results)


def main():
    """
    Main function of the script. It parses the arguments and runs the NN.
    """
    
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument("-b", "--num-batch", type=int, default=32,
                        help="Number of batches")
    parser.add_argument("-p", "--image-path", type=str,
                        help="Path to test image")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of Epochs")
    parser.add_argument("-lr", "--learning-rate", type=float, default=2e-3,
                        help="Leaning rate of the Net")
    parser.add_argument("--gamma", type=float, default=0.7,
                        help="Leaning rate step")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Compile the model")
    parser.add_argument("--load-model", action="store_true", default=False,
                        help="Load a pre-trained model")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="Save after training")
    parser.add_argument("-v", '--verbose', action="store_true", default=False,
                        help='Prints everything')

    args = parser.parse_args()

    if args.verbose:
        ic.enable()
    else:
        ic.disable()

    train_loader, test_loader = get_dataset(args.num_batch)

    if args.load_model:
        model = Net_load()  # because the model loaded has been trained with less layer than the one I've built
        model = load_model(model, args.compile1)

    else:
        model = Net()
        model = model.to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate)

        if args.compile:
            model = torch.compile(model)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
            val(model, device, test_loader)
            scheduler.step()

    if args.image_path is not None:
        try:
            image = preprocess(args.image_path).to(device)
        except Exception as e:
            print(f"Error preprocessin the image: {e}")
        result = infer(model, device, image)
        result = postprocess(result)
        print(f"Result for the image '{args.image_path}': {str(result)}")

    if args.save_model:
        try:
            torch.save(model.state_dict(), "./model/mnist_cnn.pt")  # pt, safer than pth
        except Exception as e:
            print(f"Error {e}")


if __name__ == '__main__':
    main()
