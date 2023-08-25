import torch
import os
import argparse
import datetime
import warnings
import numpy as np
import random
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
from classifier import *
from util_functions import get_data_subsampler
from dataset import mnist_dataset_with_z

warnings.filterwarnings('ignore')

## fix random sampling
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## create parser
parser = argparse.ArgumentParser(
    description="train generic models", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--train_type", default="normal", choices=["normal", "proximal", "admm"], type=str, dest="train_type")
parser.add_argument("--model", default="linear_classifier", choices=["linear_classifier", "basic_classifier"], type=str, dest="model")
parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"], type=str, dest="dataset")
parser.add_argument("--optimizer", default="sgd", choices=["adam", "adadelta", "sgd"], type=str, dest="optimizer")
parser.add_argument("--front_loss", default="l2", choices=["l1", "l2"], type=str, dest='front_loss')
parser.add_argument("--use_subset", default=False, type=bool, dest="use_subset")
parser.add_argument("--data_per_class", default=5000, type=int, dest="data_per_class")
parser.add_argument("--bottleneck_pos", default=2, choices=[0, 1, 2], type=int, dest="bottleneck_pos")

args = parser.parse_args()

train_type = args.train_type
model = args.model
dataset = args.dataset
optimizer = args.optimizer
front_loss = args.front_loss
use_subset = args.use_subset
data_per_class = args.data_per_class
bottleneck_pos = args.bottleneck_pos

## result path
date_and_time = datetime.datetime.now()
title = model + "_" + train_type + "_" + optimizer + "_" + dataset + "_" + date_and_time.strftime('%Y-%m-%d_%H-%M-%S')
result_dir = "./results"

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

result_path = os.path.join(result_dir, title)

if not os.path.exists(result_path):
    os.mkdir(result_path)

## hyperparameters
## classifier
if "classifier" in model:
    number_epoch = 200
    if "linear" in model:
        img_size = 28
    else:
        img_size = 32
    batch_size = 64

    if train_type == "normal":
        ## learning_rate_front, learning_rate_back
        learning_rate = [0.013, 0.013]
    elif train_type == "proximal":
        ## learning_rate_front, learning_rate_back, learning_rate_z, lambda, z_iter
        learning_rate = [0.013, 0.013, 0.5, 30, 80]
    elif train_type == "admm":
        ## learning_rate_front, learning_rate_back, learning_rate_z, learning_rate_y, lambda, z_iter
        learning_rate = [0.013, 0.013, 0.2, 1, 30, 50]

else:
    raise NotImplementedError

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## load dataset
if "classifier" in model:
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
        ])
        dataset_path = "/hdd1/dataset/"
        dataset_train = MNIST(dataset_path, train=True, download=False, transform=transform)
        dataset_test = MNIST(dataset_path, train=False, download=False, transform=transform)

        if train_type == "proximal" or train_type == "admm":
            dataset_train = mnist_dataset_with_z(dataset_train)

        img_channel = 1

    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        dataset_path = "/hdd1/dataset/"
        dataset_train = CIFAR10(dataset_path, train=True, download=False, transform=transform)
        dataset_test = CIFAR10(dataset_path, train=False, download=False, transform=transform)
        img_channel = 3
    else:
        raise NotImplementedError

    ## make dataloaders
    if use_subset:
        subsampler = get_data_subsampler(dataset_train, data_per_class)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=subsampler)
    else:
        if train_type == "proximal" or train_type == "admm":
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=dataset_train.collate_fn)
        else:
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

else:
    raise NotImplementedError

## train configurations
config = {
    "model": model,
    "train_type": train_type,
    "bottleneck_pos": bottleneck_pos,
    "optimizer": optimizer,
    "front_loss": front_loss, 
    "number_epoch": number_epoch,
    "img_size": img_size,
    "img_channel": img_channel,
    "learning_rate": learning_rate,
    "dataloader_train": dataloader_train,
    "dataloader_test": dataloader_test,
    "result_path": result_path,
    "device": device
}

config_path = os.path.join(result_path, "config.txt")

with open(config_path, "w") as f:
    f.write("Model: " + str(model) + "\n")
    f.write("Train type: " + str(train_type) + "\n")
    f.write("Bottleneck position: " + str(bottleneck_pos) + "\n")
    f.write("Epoch: " + str(number_epoch) + "\n")
    f.write("Channel: " + str(img_channel) + "\n")
    f.write("Dataset: " + str(dataset) + "\n")
    f.write("Optimizer: " + str(optimizer) + "\n")
    f.write("Front loss: " + str(front_loss) + "\n")
    if use_subset:
        f.write("Subset size: " + str(data_per_class) + "\n")
    f.write("Learning rate: " + str(learning_rate) + "\n")

## training
if "classifier" in model:
    if train_type == "normal":
        train_normal_classifier(config)
    elif train_type == "proximal":
        train_proximal_classifier(config)
    elif train_type == "admm":
        train_admm_classifier(config)
    else:
        raise NotImplementedError

else:
    raise NotImplementedError