## Food Classification
### Import Libraries
```
import os
import numpy as np
import time
import random
import copy
import pdb 
from tqdm import tqdm
from functools import partial 
from utils import * 
import argparse
import json 
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.model_zoo as model_zoo
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from torch.autograd.function import Function
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from __future__ import print_function, division

print(os.getcwd()) 
data_path = '/home/Food59/' 

train_batchsize = 8
train_dir = '/home/Food59/train'
validation_dir = '/home/Food59/val'

train_datagen = ImageDataGenerator(rescale=1./255,
    horizontal_flip=True,
    validation_split=0.2) 
```
```
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=train_batchsize,
    class_mode='categorical',
    subset='training') 
```
```
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        feat = self.avgpool(self.layer4(x))
        feat = feat.view(feat.size(0), -1)
        x = self.fc(feat)
        
        return feat, x   

def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
```
### Data Transformation
```
def transfrom_data():
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])
    }

    return data_transforms
```
### Loading the data with transformation
```
def load_data(batch_size, num_workers):
    print("Start loading data")
    data_dir = '/home/Food59/'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transfrom_data()[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) \
                    for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print("Dataset sizes: Train {} Val {}".format(dataset_sizes['train'], dataset_sizes['val']))
    print("Number of classes: Train {} Val {}".format(len(image_datasets['train'].classes), len(image_datasets['val'].classes)))

    return dataloaders, class_names, dataset_sizes
```
### Loading the model
```
def load_model(class_names):
    model = resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names)) 

    model = torch.nn.DataParallel(model.cuda(), device_ids=[0])

    return model
```
```
class CenterLoss(nn.Module):
    def __init__(self, num_classes=59, feat_dim=512, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
 ```
 ```
 def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append((int(correct_k), correct_k.mul_(1.0 / batch_size)))
        return res
```
```
def train_model(dataloaders, model, dataset_sizes, criterion, optimizer, num_epochs, save_dir, f):
    since = time.time()
    
    best_val_top1_acc = 0.0
    best_val_epoch = -1 
    final_val_top5_acc = 0.0

    for epoch in range(num_epochs):  
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            top1_running_corrects = 0
            top5_running_corrects = 0

            it = tqdm(range(len(dataloaders[phase])), desc="Epoch {}/{}, Split {}".format(epoch, num_epochs - 1, phase), ncols=0)
            data_iter = iter(dataloaders[phase])
            for niter in it:
                inputs, labels = data_iter.next()
                inputs = inputs.cuda()
                labels = labels.cuda()

                optimizer[0].zero_grad()
                optimizer[1].zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    feats, outputs = model(inputs)
                    
                    loss = centerloss(feats, labels) * loss_weight + nllloss(outputs, labels)
                    loss = nllloss(outputs, labels)
                    prec1, prec5 = accuracy(outputs, labels, topk=(1,5))

                    if phase == 'train':
                        loss.backward()

                        optimizer[0].step()
                        optimizer[1].step()

                training_loss = loss.item()
                running_loss += loss.item() * inputs.size(0)
                top1_running_corrects += prec1[0]
                top5_running_corrects += prec5[0]

            epoch_loss = running_loss / dataset_sizes[phase]
            top1_epoch_acc = float(top1_running_corrects) / dataset_sizes[phase]
            top5_epoch_acc = float(top5_running_corrects) / dataset_sizes[phase]
            print('{} Epoch Loss: {:.6f} Epoch top1 Acc: {:.6f} Epoch top5 Acc: {:.6f}\n'.format(phase, epoch_loss, top1_epoch_acc, top5_epoch_acc))
            with open(epoch_trace_f_dir, "a") as f:
                lr = optimizer[0].param_groups[0]['lr']
                f.write("{},{},{},{:e},{:e},{:e}\n".format(epoch,phase,lr,epoch_loss,top1_epoch_acc,top5_epoch_acc))

            if phase == 'val' and top1_epoch_acc > best_val_top1_acc:
                print("Top1 val Acc improve from {:6f} --> {:6f}".format(best_val_top1_acc, top1_epoch_acc))
                best_val_top1_acc = top1_epoch_acc
                final_val_top5_acc = top5_epoch_acc
                best_val_epoch = epoch
                save_f_dir = os.path.join(save_dir, "best_val_model.ft")
                print("Saving best val model into {}...".format(save_f_dir))
                torch.save(model.state_dict(), save_f_dir)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best top1 val Acc: {:6f}'.format(best_val_top1_acc))
    print('Final top5 val Acc: {:6f}'.format(final_val_top5_acc))
    print('Best val model is saved at epoch # {}'.format(best_val_epoch))
```
```
if __name__=="__main__":

    dataloaders, class_names, dataset_sizes = load_data(32, 12)
    model= load_model(class_names)

    nllloss = nn.CrossEntropyLoss()
    
    loss_weight = 0.001
    centerloss = CenterLoss(num_classes=59, feat_dim=512, use_gpu=True)
    
    nllloss = nllloss.cuda()
    centerloss = centerloss.cuda()
    model = model.cuda()
    
    criterion = [nllloss,centerloss]
    
    optimizer4nn = optim.Adagrad(model.parameters(), lr=0.001) 

    optimizer4center = optim.Adagrad(centerloss.parameters(), lr=0.001)
    optimizer = [optimizer4nn,optimizer4center]
      
    num_epochs = 20

    save_dir = './outputs/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    epoch_trace_f_dir = os.path.join(save_dir, "trace.csv")
    with open(epoch_trace_f_dir, "w") as f:
        f.write("epoch,split,lr,loss,top1_acc,top5_acc\n")

    train_model(dataloaders, model, dataset_sizes, criterion, optimizer, num_epochs, save_dir, f)
```
Training at 20 epochs results in best validation accuracy of >98%.
