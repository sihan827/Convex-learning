import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from torch.utils.data import SubsetRandomSampler


def plot_results_per_epoch(result_mean, result_std, x_label, y_label, title, path):
    
    plt.figure(figsize=(8, 6))
    plt.title(title)
    
    plt.plot(range(len(result_mean)), result_mean, "-", color='red')
    if result_std is not None:
        plt.fill_between(
            range(len(result_mean)), result_mean - result_std, result_mean + result_std, facecolor="blue", alpha=0.3
        )
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    plt.tight_layout()
    plt.show()
    
    full_path = os.path.join(path, title)
    plt.savefig(full_path)
    plt.clf()
    plt.close()


def plot_img_per_epoch(img, title, path):
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.show()

    full_path = os.path.join(path, title)
    plt.savefig(full_path)
    plt.clf()
    plt.close()


def get_log(log_path):
    
    train_type = log_path.split('_')[2]
    if train_type == 'core':
        front_train_loss_mean = []
        front_train_loss_std = []
    back_train_loss_mean = []
    back_train_loss_std = []
    train_acc_mean = []
    train_acc_std = []
    back_test_loss_mean = []
    back_test_loss_std = []
    back_test_acc_mean = []
    back_test_acc_std = []

    with open(log_path, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.split(',')
            if train_type == 'core':
                front_train_loss_mean.append(float(line[1]))
                front_train_loss_std.append(float(line[2]))
                back_train_loss_mean.append(float(line[3]))
                back_train_loss_std.append(float(line[4]))
                train_acc_mean.append(float(line[5]))
                train_acc_std.append(float(line[6]))
                back_test_loss_mean.append(float(line[7]))
                back_test_loss_std.append(float(line[8]))
                back_test_acc_mean.append(float(line[9]))
                back_test_acc_std.append(float(line[10]))
            else:
                back_train_loss_mean.append(float(line[1]))
                back_train_loss_std.append(float(line[2]))
                train_acc_mean.append(float(line[3]))
                train_acc_std.append(float(line[4]))
                back_test_loss_mean.append(float(line[5]))
                back_test_loss_std.append(float(line[6]))
                back_test_acc_mean.append(float(line[7]))
                back_test_acc_std.append(float(line[8]))
    
    if train_type == 'core':
        logs = [
            np.array(front_train_loss_mean),
            np.array(front_train_loss_std),
            np.array(back_train_loss_mean),
            np.array(back_train_loss_std),
            np.array(train_acc_mean), 
            np.array(train_acc_std), 
            np.array(back_test_loss_mean), 
            np.array(back_test_loss_std),
            np.array(back_test_acc_mean),
            np.array(back_test_acc_std)
        ]
        return logs
    else:
        logs = [
            np.array(back_train_loss_mean),
            np.array(back_train_loss_std),
            np.array(train_acc_mean), 
            np.array(train_acc_std), 
            np.array(back_test_loss_mean), 
            np.array(back_test_loss_std),
            np.array(back_test_acc_mean),
            np.array(back_test_acc_std)
        ]
        return logs


def plot_logs(logs, type, label, title, xlim=None, ylim=None, plot_std=True):
    if type == "back train loss":
        i, j, k, l = 2, 3, 0, 1
        normal = logs[-1]
        logs = logs[:-1]
    elif type == "front train loss":
        i, j = 0, 1
        logs = logs[:-1]
    elif type == "train accuracy":
        i, j, k, l = 4, 5, 2, 3 
        normal = logs[-1]
        logs = logs[:-1]
    elif type == "back test loss":
        i, j, k, l = 6, 7, 4, 5
        normal = logs[-1]
        logs = logs[:-1]
    elif type == "test accuracy":
        i, j, k, l = 8, 9, 6, 7
        normal = logs[-1]
        logs = logs[:-1]
    plt.figure(figsize=(8, 8), dpi=600)
    plt.title(title)
    x = np.arange(len(logs[0][1]))
    if xlim == None:
        tick_rate = 25
    elif xlim[1] >= 50:
        tick_rate = 25
    else:
        tick_rate = 10
    x_tick = x[::tick_rate]
    for a, log in enumerate(logs):
        plt.plot(x, log[i], label=str(label[a]))
    if plot_std:
        for a, log in enumerate(logs):
            plt.fill_between(x, log[i] - log[j], log[i] + log[j], alpha=0.3)
    if type != "front train loss":
        plt.plot(x, normal[k], label=str(label[-1]), color='k')
        if plot_std:
            plt.fill_between(x, normal[k] - normal[l], normal[k] + normal[l], alpha=0.3)
    plt.xlabel('iteration')
    plt.ylabel(type)
    plt.xticks(x_tick)
    if xlim != None and ylim != None:
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
    plt.legend()
    plt.show()


def get_data_subsampler(dataset, data_per_class):
    assert data_per_class < len(dataset) / len(dataset.classes)
    subset_idx = []
    
    for label in dataset.targets.unique():
        class_idx = (dataset.targets == label).nonzero(as_tuple=True)
        rand_select = torch.randint(0, len(class_idx[0]), (data_per_class, ))
        class_idx = class_idx[0][rand_select]
        subset_idx.append(class_idx)
    
    subset_idx = torch.cat(subset_idx)
    
    return SubsetRandomSampler(indices=subset_idx)


def compute_psnr(loss):
    return -10. * np.log(loss) / np.log(10.)


def compute_accuracy(prediction, label):
    
    correct = torch.argmax(prediction, axis=1) == label
    accuracy = torch.mean(correct.type(torch.cuda.FloatTensor)).detach().cpu().numpy()

    return accuracy