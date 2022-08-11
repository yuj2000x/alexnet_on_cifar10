# 更改參數去符合 CIFAR10
# GPU 版本

from re import T
import torch
import torchvision.transforms as transforms
from torch import nn
from torchvision.transforms import ToTensor, Lambda
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 使用 mps
device = torch.device('mps')

# 定義全域變數
numPrint = 20

# 定義 learning history
writer = SummaryWriter('./Result/ex-gpu-1')

# 定義參數
batch_size = 512
learning_rate = 1e-1
momentum = 0.9
epoch = 25

# 資料處理
transform_train = transforms.Compose([
    transforms.Resize((70, 70)),
    transforms.RandomCrop((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])
transform_test = transforms.Compose([
    transforms.Resize((70, 70)),
    transforms.RandomCrop((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])

# 下載 CIFAR 10
trainset = datasets.CIFAR10(
    root='./CIFAR10_data', train=True, download=False, transform=transform_train)
testset = datasets.CIFAR10(
    root='./CIFAR10_data', train=False, download=False, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, pin_memory=True)
testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, pin_memory=True)

NUM_classes = len(trainset.classes)
print('number of class: %d' % NUM_classes)
print('==>>> total training batch number: {}'.format(len(trainloader)))
print('==>>> total testing batch number: {}'.format(len(testloader)))
print('========================================')


# 定義 network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            # input size = 64 * 64  * 3
            # CNN_1: ( (64-11+2*2)/4 ) + 1 = 15
            nn.Conv2d(3, 64, 11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # 15/2 = 7
            # CNN_2: 7-5+2*2+1 = 7
            nn.Conv2d(64, 192, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # 7/2 = 3
            # CNN_3: 3-3+2+1 = 3
            nn.Conv2d(192, 384, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # CNN_4: 3-3+2+1 = 3
            nn.Conv2d(384, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # CNN_5: 3-3+2+1 = 3
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # output=(1,1,256)
            # FC_1
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256*1*1, 4096),
            nn.ReLU(inplace=True),
            # FC_2
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # FC_3
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        return self.model(x)


model = Net().to(device)


# 定義訓練函數
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0.0
    for batch, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # 將梯度至零
        pred = model(images)
        loss = loss_fn(pred, labels).to(device)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()        # 反向傳播
        optimizer.step()       # 優化

        if batch % numPrint == 0:   # 每 numPrint 個 batch 顯示一次
            loss, current = loss.item(), batch * len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= num_batches
    return train_loss


# 定義測試結果
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            loss = loss_fn(pred, labels)
            test_loss += loss.item()
            correct += (pred.argmax(1) ==
                        labels).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, 100*correct


# 定義 loss function 和 optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, momentum=momentum)


def write(train_loss, test_loss, Acc, T):
    writer.add_scalars('Training Loss : ', {
        'Training': train_loss}, T)
    writer.add_scalars('Testing Loss : ', {
        'Testing': test_loss}, T)
    writer.add_scalars('Accuracy : ', {
        'Accuracy': Acc}, T)


for t in range(epoch):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss = train_loop(trainloader, model, loss_fn, optimizer)
    test_loss, Acc = test_loop(testloader, model, loss_fn)
    T = t + 1
    write(train_loss, test_loss, Acc, T)
print("Done!")
