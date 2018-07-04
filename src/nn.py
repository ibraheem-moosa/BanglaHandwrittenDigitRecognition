import torch
from torch import nn
from torch import optim
import torchvision
import numpy as np
from skimage.io import imread
import sys
import time
import PIL
from NeuralNet import LeNet


if __name__ == '__main__':
    
    if len(sys.argv) < 4:
        print('Usage: python3 nn.py train_dir validation_dir test_dir')
        print('Example: python3 nn.py data/A/BW_50x50/train data/A/BW_50x50/validation data/A/BW_50x50/test ')
        print('Run this after prepare_dataset.py')
        print('Validation accuracy is printed in each epoch.')
        print('If it does not get better for 30 epochs, training stops')
        print('We use LeNet which is defined in NeuralNet.py.')
        print('Also training images are randomly rotated by 0.5 degree, randomly translated by 10% while training')
        sys.exit()

    train_dataset_dir = sys.argv[1]
    validation_dataset_dir = sys.argv[2]
    test_dataset_dir = sys.argv[3]
    batch_size = 4
    transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(),
                                                torchvision.transforms.RandomAffine(0.5, 
                                                                                    translate=(0.1,0.1), 
                                                                                    resample= PIL.Image.BILINEAR),
                                                torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder(train_dataset_dir, transform=transform)
    validation_transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()])
    validation_dataset = torchvision.datasets.ImageFolder(validation_dataset_dir, transform=transform)
    print(dataset.class_to_idx)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1024, num_workers=4)
    image_size = 50 * 50
    net = LeNet()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.90)
    criterion = nn.CrossEntropyLoss()

    best_acc_seen = 0
    best_acc_seen_at = 0
    for epoch in range(1, 1000):
        print(time.strftime('%Y-%m-%d %H:%M'))
        running_loss = 0.0
        img_count = 0
        correct = 0
        for img, target in dataloader:
            img_count += len(target)
            target = torch.tensor(target)
            optimizer.zero_grad()
            #img = img.reshape((-1, image_size)).to(torch.float32)
            pred = net.forward(img)
            loss = criterion(pred, target)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(pred.data, 1)
            correct += (predicted == target).sum().item()
        accuracy = correct / img_count
        print('Epoch: {}\nLoss: {:02f}\nTra Accuracy: {:02f}, Mistakes: {}'.format(epoch, running_loss/img_count, accuracy, img_count - correct))
        scheduler.step()
        net.eval()
        img_count = 0
        correct = 0
        for img, target in validation_dataloader:
            img_count += len(target)
            target = torch.tensor(target)
            pred = net.forward(img)
            _, predicted = torch.max(pred.data, 1)
            correct += (predicted == target).sum().item()
        accuracy = correct / img_count
        print('Val Accuracy: {:02f}, Mistake: {}, Best: {} seen at {}'.format(accuracy, img_count - correct, best_acc_seen, best_acc_seen_at))
        if accuracy >= best_acc_seen:
            best_acc_seen = accuracy
            best_acc_seen_at = epoch
        if epoch - best_acc_seen_at > 30:
            print(best_acc_seen)
            break

        net.train()

    test_dataset = torchvision.datasets.ImageFolder(test_dataset_dir, transform=validation_transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, num_workers=4)
    net.eval()
    img_count = 0
    correct = 0
    for img, target in test_dataloader:
        img_count += len(target)
        target = torch.tensor(target)
        pred = net.forward(img)
        _, predicted = torch.max(pred.data, 1)
        correct += (predicted == target).sum().item()
    accuracy = correct / img_count
    print('Tes Accuracy: {}'.format(accuracy))

    torch.save(net.state_dict(), 'LeNet')
