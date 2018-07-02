import torch
from torch import nn
from torch import optim
from torch.nn.functional import relu
from torch.nn.functional import leaky_relu
from torch.nn.functional import dropout
import torchvision
import numpy as np
from skimage.io import imread
import sys
import time

class FeedForward(nn.Module):
    def __init__(self, input_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 10)
        self.lsm = nn.LogSoftmax(dim=0)
        self._init_weights_()

    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = self.fc4(x)
        return self.lsm(x)
    
    def _init_weights_(self):
        gain = nn.init.calculate_gain('relu')
        #gain = 10
        init_method = nn.init.xavier_normal_
        bias_init_method = nn.init.constant_
        init_method(self.fc1.weight.data, gain=gain)
        init_method(self.fc2.weight.data, gain=gain)
        init_method(self.fc3.weight.data, gain=gain)
        init_method(self.fc4.weight.data, gain=gain)
        bias_init_method(self.fc1.bias.data, 0)
        bias_init_method(self.fc2.bias.data, 0)
        bias_init_method(self.fc3.bias.data, 0)
        bias_init_method(self.fc4.bias.data, 0)


if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print('Usage: python3 nn.py dataset_directory')

    batch_size = 512
    transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder(sys.argv[1],transform=transform)
    print(dataset.class_to_idx)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    image_size = 30 * 30
    net = FeedForward(image_size)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    criterion = nn.NLLLoss()

    print(time.strftime('%Y-%m-%d %H:%M'))
    best_acc_seen = 0
    best_acc_seen_at = 0
    for epoch in range(1, 1000):
        running_loss = 0.0
        img_count = 0
        correct = 0
        for img, target in dataloader:
            img_count += len(target)
            target = torch.tensor(target)
            optimizer.zero_grad()
            img = img.reshape((-1, image_size)).to(torch.float32)
            pred = net.forward(img)
            loss = criterion(pred, target)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(pred.data, 1)
            correct += (predicted == target).sum().item()
        accuracy = correct / img_count
        print('Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, running_loss/img_count, accuracy))
        print(time.strftime('%Y-%m-%d %H:%M'))
        if accuracy > best_acc_seen:
            best_acc_seen = accuracy
            best_acc_seen_at = epoch
        if epoch - best_acc_seen_at > 20:
            print(best_acc_seen)
            break
