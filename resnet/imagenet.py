import torch, os
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from train_model import train_model
from test_model import test_model
from torch.utils.data import TensorDataset, DataLoader
import time
import sys

class CNN(nn.Module):
    def __init__(self, input_shape=1, mid_dim=256, num_classes=10):
        try:
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_shape,
                          32,
                          kernel_size=5,
                          padding=1,
                          stride=1,
                          bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3, 3))
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(32,
                          64,
                          kernel_size=5,
                          padding=1,
                          stride=1,
                          bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3, 3))
            )
            self.fc1 = nn.Sequential(
                nn.Linear(36864, 512),
                nn.ReLU(inplace=True)
            )
            self.fc = nn.Linear(512, num_classes)
        except Exception as e:
            print("CNN")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            out = self.conv1(x)
            out = self.conv2(out)
            out = torch.flatten(out, 1)
            out = self.fc1(out)
            out = self.fc(out)
            return out
        except Exception as e:
            print("CNN forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def dataset(data_path):
    """Load ImageNet (training and val set)."""

    # Load ImageNet and normalize
    traindir = os.path.join(data_path, "train")
    valdir = os.path.join(data_path, "val")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    trainLoader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    testLoader = DataLoader(dataset=val_dataset, batch_size=256, shuffle=False)

    return trainLoader, testLoader


data_dir = '/home/claudio/Documentos/pycharm_projects/FL-H.IAAC/dataset_utils/data/Tiny-ImageNet/raw_data/tiny-imagenet-200'
# data_dir = '/home/claudio/FL-H.IAAC/dataset_utils/data/Tiny-ImageNet/raw_data/tiny-imagenet-200'

loss_ft = nn.CrossEntropyLoss()
trainloader, testloader = dataset(data_dir)

#Load Resnet18
model_ft = CNN(input_shape=3, mid_dim=400, num_classes=200)
#Finetune Final few layers to adjust for tiny imagenet input
# model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
# model_ft.fc = torch.nn.Linear(in_features=512, out_features=200, bias=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Dispositivo: ", device)
model_ft = model_ft.to(device)
#Loss Function
criterion = nn.CrossEntropyLoss().to(device)
# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.01)
train_loss = 0
train_acc = 0
# train_model7("48",model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, num_epochs=10)
train_num = 0
log_interval = 10
for step in range(1):
    start_time = time.process_time()
    for i, (x, y) in enumerate(trainloader):
        if type(x) == type([]):
            x[0] = x[0].to(device)
        else:
            x = x.to(device)
        y = y.to(device)
        train_num += y.shape[0]

        optimizer_ft.zero_grad()
        output = model_ft(x)
        # y = torch.tensor(y.int().detach())
        loss = loss_ft(output, y)
        train_loss += loss.item() * y.shape[0]
        loss.backward()
        optimizer_ft.step()

        train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        # train_acc += torch.sum(output == y.data)

        if i % log_interval == 0:
            total_time = time.process_time() - start_time
            print('Train Epoch: {} [{}]\tLoss: {:.6f}\t Acc: {}'.format(
                step, (i+1) * len(x), loss.item(), train_acc/train_num))
            print("Duração: ", total_time)
            start_time = time.process_time()

avg_loss_train = train_loss / train_num
avg_acc_train = train_acc / train_num

print("Acc: ", train_acc, " loss: ", avg_loss_train)
