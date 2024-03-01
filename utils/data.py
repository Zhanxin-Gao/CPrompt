import os
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from PIL import Image
import torch

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

class iDomainnetCIL(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0)),
    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        rootdir = "[DATA-PATH]"
        train_txt = './data/datautils/domainnet/train.txt'
        test_txt = './data/datautils/domainnet/test.txt'

        train_images = []
        train_labels = []
        with open(train_txt, 'r') as dict_file:
            for line in dict_file:
                (value, key) = line.strip().split(' ')
                train_images.append(os.path.join(rootdir, value))
                train_labels.append(int(key))
        
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        test_images = []
        test_labels = []
        with open(test_txt, 'r') as dict_file:
            for line in dict_file:
                (value, key) = line.strip().split(' ')
                test_images.append(os.path.join(rootdir, value))
                test_labels.append(int(key))
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)

        self.train_data = train_images
        self.train_targets = train_labels
        
        self.test_data = test_images
        self.test_targets = test_labels


class iImageNetR(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]

    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0)),
    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        rootdir = "[DATA-PATH]"
        train_txt = './data/datautils/imagenet-r/coda_train.txt'
        test_txt = './data/datautils/imagenet-r/coda_test.txt'

        train_images = []
        train_labels = []
        with open(train_txt, 'r') as dict_file:
            for line in dict_file:
                
                (value, key) = line.strip().split(' ')
                train_images.append(os.path.join(rootdir, value))
                train_labels.append(int(key))
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)

        test_images = []
        test_labels = []
        with open(test_txt, 'r') as dict_file:
            for line in dict_file:
                (value, key) = line.strip().split(' ')
                test_images.append(os.path.join(rootdir, value))
                test_labels.append(int(key))
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)

        self.train_data = train_images
        self.train_targets = train_labels
        self.test_data = test_images
        self.test_targets = test_labels
        

class iCIFAR100_vit(iData):
    use_path = False
    train_trsf = [
        transforms.Resize(256),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(100).tolist()
   
    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100('./data', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100('./data', train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)
        

class iStanford_cars(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]

    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0)),
    ]

    class_order = np.arange(45).tolist()

    def download_data(self):
        rootdir = "[DATA-PATH]"
        train_txt = './data/datautils/stanfordcars/train.txt'
        test_txt = './data/datautils/stanfordcars/test.txt'
        
        train_images = []
        train_labels = []
        with open(train_txt, 'r') as dict_file:
            for line in dict_file:
                # (key, value) = line.strip().split('\t')
                (value, key) = line.strip().split(' ')
                train_images.append(os.path.join(rootdir, value))
                train_labels.append(int(key))
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)

        test_images = []
        test_labels = []
        with open(test_txt, 'r') as dict_file:
            for line in dict_file:
                # (key, value) = line.strip().split('\t')
                (value, key) = line.strip().split(' ')
                test_images.append(os.path.join(rootdir, value))
                test_labels.append(int(key))
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)

        self.train_data = train_images
        self.train_targets = train_labels
        self.test_data = test_images
        self.test_targets = test_labels
