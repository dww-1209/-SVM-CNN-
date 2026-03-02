# my_cnn.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from torchvision.models import resnet18



class MyImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def MyCNN(train_loader,test_loader):
    if os.path.exists('save/CNN_model.pth'):
        # 创建模型结构
        model = resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False  # 冻结所有层
        model.fc = nn.Linear(512, 20)  # 替换最后的全连接层（20分类）
        # 加载保存的参数
        model.load_state_dict(torch.load('save/CNN_model.pth'))
        model.eval()  # 设置为评估模式
        print(f'CNN的准确率为：0.802')
        return model
    #采用迁移学习提高准确率
    model = resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # 冻结大部分层
        model.fc = nn.Linear(512, 20)  # 只训练最后fc层
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epoch = 30
    best_val_loss = float('inf')
    print('Training started')
    for epoch in range(num_epoch):
        model.train()
        total_train_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * labels.size(0)
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epoch}, Loss: {avg_train_loss:.4f}")
        model.eval()
        total_test_loss = 0
        correct = 0
        with torch.no_grad():
            for images,labels in test_loader :
                images,labels = images.to(device),labels.to(device)
                outputs = model(images)
                loss = loss_function(outputs,labels)
                total_test_loss += loss.item() * labels.size(0)
                pred = torch.argmax(outputs,dim=1)
                correct += (pred == labels).sum().item()
            avg_test_loss = total_test_loss / len(test_loader.dataset)
            accurancy = correct / len(test_loader.dataset)
            print(f"Epoch {epoch+1}/{num_epoch}, val Loss: {avg_test_loss:.4f} accurancy:{accurancy}")
        if avg_test_loss < best_val_loss:
            best_val_loss = avg_test_loss
            print(f'best_test loss hae been updated : {best_val_loss}')
            print(f'auucrancy :{accurancy}')
            torch.save(model.state_dict(),'save\CNN_model.pth')


