import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device =='cuda':
    torch.cuda.manual_seed_all(777)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

'''
데이터셋
'''
trainset = torchvision.datasets.CIFAR10(
    root = './cifar10',
    train = True,
    download = True,
    transform = transform,
)

testset = torchvision.datasets.CIFAR10(
    root = './cifar10',
    train = False,
    download = True,
    transform = transform
)

'''
데이터로더
'''
train_loader = torch.utils.data.DataLoader(
    dataset = trainset,
    batch_size = 512,
    shuffle = True,
    num_workers = 0
)

test_loader = torch.utils.data.DataLoader(
    dataset = testset,
    batch_size = 4,
    shuffle = False,
    num_workers = 0
)

'''
클래스
'''
classes = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

'''
이미지 함수
'''
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


'''
모델
'''
import torchvision.models.vgg as vgg

cfg = [32,32,'M', 64,64,128,128,128,'M',256,256,256,512,512,512,'M'] #13 + 3 =vgg16

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

'''
모델 정의
'''
vgg16 = VGG(vgg.make_layers(cfg),10,True).to(device)

'''
모델 테스트
'''
a = torch.Tensor(1, 3, 32, 32).to(device)
out = vgg16(a)
print(out)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(vgg16.parameters(), lr = 0.005, momentum = 0.9)
lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.9)

'''
모델 훈련
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
        hypothesis = vgg16(x_train)
        loss = criterion(hypothesis, y_train)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print('Epoch: {}/{} | Train Loss: {}'.format(epoch + 1, epochs, train_loss))

'''
모델 평가
'''
dataiter = iter(test_loader)
x_test, y_test = dataiter.next()

imshow(torchvision.utils.make_grid(x_test))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

output = vgg16(x_test.to(device))
_, predicted = torch.max(output, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0

with torch.no_grad():
    for samples in test_loader:
        x_test, y_test = samples
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        predicted = vgg16(x_train)

        _, predicted = torch.max(output.data, 1)

        correct += y_test.size(0)
print('정확도: {}'.format(100 * correct / total`))