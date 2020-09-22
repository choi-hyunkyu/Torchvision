import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device =='cuda':
    torch.cuda.manual_seed_all(777)

'''
메타데이터
'''
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(
    root = './cifar10',
    train = True,
    download = True,
    transform = transform
)

'''
mean, normalization
'''
train_data_mean = trainset.data.mean(axis = (0, 1, 2))
train_data_mean = train_data_mean / 255
train_data_std = trainset.data.std(axis = (0, 1, 2))
train_data_std = train_data_std / 255

'''
데이터셋
'''
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding = 4),
    transforms.ToTensor(),
    transforms.Normalize(train_data_mean, train_data_std)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_data_mean, train_data_std)
])

trainset = torchvision.datasets.CIFAR10(
    root = './cifar10',
    train = True,
    download = True,
    transform = transform
)

testset = torchvision.datasets.CIFAR10(
    root = './cifar10',
    train = False,
    download = True,
    transform = transform
)

train_loader = torch.utils.data.DataLoader(
    dataset = trainset,
    batch_size = 256,
    shuffle = True,
    num_workers = 0
)

test_loader = torch.utils.data.DataLoader(
    dataset = testset,
    batch_size = 256,
    shuffle = False,
    num_workers = 0
)

'''
모델 설계
'''
import torchvision.models.resnet as resnet

conv1x1 = resnet.conv1x1
Bottleneck = resnet.Bottleneck
BasicBlock = resnet.BasicBlock

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        #x.shape =[1, 16, 32,32]
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        #x.shape =[1, 128, 32,32]
        x = self.layer2(x)
        #x.shape =[1, 256, 32,32]
        x = self.layer3(x)
        #x.shape =[1, 512, 16,16]
        x = self.layer4(x)
        #x.shape =[1, 1024, 8,8]
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

resnet50 = ResNet(resnet.Bottleneck, [3, 4, 6, 3], 10, True).to(device)

'''
모델 테스트
'''
a = torch.Tensor(1, 3, 32, 32).to(device)
output = resnet50(a)
output

'''
criterion, optimizer lr_sche 정의
'''
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(resnet50.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 5e-4)
lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.5)

'''
정확도
'''
def acc_check(model, test_set, epoch, save = 1):
    correct = 0
    total = 0
    with torch.no_grad():
        for samples in test_set:
            x_test, y_test = samples
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            hypothesis = model(x_test)
            _, predicted = torch.max(hypothesis.data, 1)

            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()

        acc = (100 * correct / total)
        print("정확도: {}".format(acc))
        if save:
            torch.save(model.state_dict(), './data/model_epoch{}_acc_{}.pth'.format(epoch, int(acc)))

        return acc

'''
학습
'''
print(len(train_loader))
epochs = 10

for epoch in range(epochs):
    train_loss = 0.0
    lr_sche.step()

    for batch_idx, samples in enumerate(train_loader, 0):
        x_train, y_train = samples
        x_train = x_train.to(device)
        y_train = y_train.to(device)

        optimizer.zero_grad()
        hypothesis = resnet50(x_train)
        loss = criterion(hypothesis, y_train)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        print("Epoch: {}/{} | Train Loss: {}".format(epoch + 1, epochs, train_loss))
    
    acc = acc_check(resnet50, test_loader, epoch, save = 1)