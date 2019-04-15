import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torchvision
import torch.utils.data as Data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import utils

num_epoches = 500
batch_size = 32
num_classes = 65
lr = 0.1
step_size = 20

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


# base_path = 'D:/data center/'
base_path = '/workspace/fubo/'
train_path = base_path + 'train/'
valid_path = base_path + 'valid/'

train_transform = transforms.Compose([
    ResizeImage(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

valid_transform = transforms.Compose([
    ResizeImage(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

img_data_source = dset.ImageFolder(root=train_path, transform=train_transform)
img_data_target = dset.ImageFolder(root=valid_path, transform=valid_transform)

# torch.utils.data.random_split(img_data_source, lengths)

loader = Data.DataLoader(
    dataset=img_data_source,  # torch TensorDataset format
    batch_size=batch_size,  # mini batch size
    shuffle=True,  # 要不要打乱数据
    num_workers=2,  # 多线程来读数据
)

testloader = Data.DataLoader(
    dataset=img_data_target,  # torch TensorDataset format
    batch_size=batch_size,  # mini batch size
    shuffle=False,  # 要不要打乱数据
    num_workers=2,  # 多线程来读数据
)


class ResNet50Fe(nn.Module):
    def __init__(self):
        super(ResNet50Fe, self).__init__()
        model_resnet50 = models.resnet50(pretrained=False)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__in_features = model_resnet50.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = ResNet50Fe()
        self.classifier = nn.Linear(self.features.output_num(), num_classes)

        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    resnet50 = ResNet50().to(device)

    params = [
        {"params": resnet50.features.parameters(), "lr": lr * 1},
        {"params": resnet50.classifier.parameters(), "lr": lr},
    ]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=0.0005, nesterov=True)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.5)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epoches):
        #     # print('epoch {}'.format(epoch + 1))
        # print('-' * 10)
        resnet50.train()
        running_loss = 0.0
        running_acc = 0.0
        for step, (batch_x, batch_y) in enumerate(loader, start=1):  # 每一步 loader 释放一小批数据用来学习
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            out = resnet50(batch_x)
            loss = loss_func(out, batch_y)
            running_loss += loss.data.item() * batch_y.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == batch_y).sum()
            running_acc += num_correct.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if step % 100 == 0:
            #     print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
            #         epoch + 1, num_epoches, running_loss / (batch_size * step),
            #         running_acc / (batch_size * step)))
        print('Finish {} epoch, Loss: {:.6f}, Accuracy of Train: {:.6f}'.format(
            epoch + 1, running_loss / (len(img_data_source)), running_acc / (len(img_data_source))))
        # print(optimizer.state_dict())
        # 打出来一些数据
        # print('Epoch: ', epoch, '| Step: ', step, '| batch x: ', batch_x.numpy(), '| batch y: ', batch_y.numpy())

        scheduler.step()
        # for param_group in optimizer.param_groups:
        #     print(param_group['lr'])

        resnet50.eval()
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = resnet50(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of Test: %.2f %%' % (100 * correct / total))

    # torch.save(squeezenet, "a-w.pkl")
