#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Author: Pengbo Song
    Modified: Yichen Nie
    Date created: 11/6/2020
    Python Version: Anaconda3 (Python 3.8.3)
'''

# Imports
import os
from collections import OrderedDict

import numpy as np
import torchvision.transforms as transforms
import cv2
import torch
import torch.utils.data as Data
import torchvision
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from torch import nn
from PIL import Image


# Average and standard deviation of MNIST dataset
# Images in MNIST dataset are grey-scale images, with only one channel
# Therefore, only one average and one std. dev. provided
mnist_avg = 0.1307
mnist_std = 0.3081


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(1, 32, kernel_size=5)),
                    ("activation1", nn.ReLU(inplace=True)),
                    ("pool1", nn.MaxPool2d(kernel_size=2, stride=2)),
                ]
            )
        )
        self.layer2 = nn.Sequential(
            OrderedDict(
                [
                    ("conv2", nn.Conv2d(32, 64, kernel_size=5)),
                    ("activation2", nn.ReLU(inplace=True)),
                    ("pool2", nn.MaxPool2d(kernel_size=2, stride=2)),
                ]
            )
        )
        self.out_layer = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(1024, 128)),
                    ("dropout1", nn.Dropout(0.2)),
                    ("activation3", nn.ReLU(inplace=True)),
                    ("fc2", nn.Linear(128, 10)),
                    ("softmax", nn.LogSoftmax(dim=1))
                ]
            )
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 1024)
        x = self.out_layer(x)
        return x

def file_name(file_dir, typ):
    files=[]
    # root = os.path.split(os.path.realpath(__file__))[0]
    for file in os.listdir(file_dir):
        if os.path.splitext(file)[1] == '.' + typ:
            files.append(file)
            # L.append(os.path.join(root, file))
    return files


# read picture
# convert to gray scale picture
# normalize to (0,1)
# normalize with average/SD from MNIST
# read label number
def readpng(filename, folder):
    img = cv2.imread(folder + filename)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_norm = np.zeros(img.shape, dtype=np.float32)
    cv2.normalize(img, img_norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    MNIST_normalizer = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((mnist_avg,), (mnist_std,))
    ])
    img_tensor = MNIST_normalizer(img_norm)

    num = int(filename[3:4])

    return img_tensor, num


def read_dataset(filename, folder):
    png_tensor = []
    label_lst = []
    for png in filename:
        pngdata, labeldata = readpng(png, folder)
        png_tensor.append(pngdata)
        label_lst.append(labeldata)
    
    png_tensor = [t.numpy() for t in png_tensor]
    png_tensor = torch.Tensor(png_tensor)
    label_lst = torch.IntTensor(label_lst).long()
    
    dataset = Data.TensorDataset(png_tensor, label_lst)
    test = Data.DataLoader(dataset=dataset, batch_size=4, shuffle=True)

    return test


def main():
    print("Load model from fullmodel.pkl.")
    model = torch.load('fullmodel.pkl')

    test_dir = '../data/numgen/'
    num_png = file_name(test_dir, 'png')

    test_set = read_dataset(num_png, test_dir)

    criterion = nn.NLLLoss(reduction='sum')

    model.eval()
    print("\nPredicting ...")
    with torch.no_grad():
        labels, pred = np.array([]), np.array([])
        loss_sum = 0.
        for data_label in test_set:
            data = data_label[0]
            label = data_label[1]
            out = model(data)
            category = np.argmax(out, axis=1)
            loss = criterion(out, label)
            loss_sum += loss
            labels = np.append(labels, label)
            pred = np.append(pred, category)
    acc = accuracy_score(labels, pred)
    cm = confusion_matrix(labels, pred) 
    rec = recall_score(labels, pred, average='macro')
    prec = precision_score(labels, pred, average='macro')

    print("Test loss = {0:.5f}".format(loss_sum))
    print("Test accuracy = {0:.5f}".format(acc))
    print("Test recall = {0:.5f}".format(rec))
    print("Test precision = {0:.5f}".format(prec))
    print("Test confusion metrix")
    print(cm)
    print('Done.')

if __name__ == '__main__':
    main()
