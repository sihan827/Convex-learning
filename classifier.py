import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from models import LinearClassifier
from models import BasicClassifier
from util_functions import plot_results_per_epoch, compute_accuracy


def train_normal_classifier(config):
    if config["model"] == "linear_classifier":
        model = LinearClassifier(
            img_size=config["img_size"], img_channel=config["img_channel"]).to(config["device"])
    elif config["model"] == "basic_classifier":
        model = BasicClassifier(
            config["img_size"], config["img_channel"], config["bottleneck_pos"]).to(config["device"])
    else:
        raise NotImplementedError

    if config["optimizer"] == "adam":
        front_optimizer = torch.optim.Adam(
            model.front_network.parameters(), lr=config["learning_rate"][0])
        back_optimizer = torch.optim.Adam(
            model.back_network.parameters(), lr=config["learning_rate"][1])
    elif config["optimizer"] == "adadelta":
        front_optimizer = torch.optim.Adadelta(model.front_network.parameters(
        ), lr=config["learning_rate"][0], rho=0.9, eps=1e-3, weight_decay=0.001)
        back_optimizer = torch.optim.Adadelta(model.back_network.parameters(
        ), lr=config["learning_rate"][1], rho=0.9, eps=1e-3, weight_decay=0.001)
    elif config["optimizer"] == "sgd":
        front_optimizer = torch.optim.SGD(model.front_network.parameters(
        ), lr=config["learning_rate"][0])
        back_optimizer = torch.optim.SGD(model.back_network.parameters(
        ), lr=config["learning_rate"][1])
    else:
        raise NotImplementedError

    criterion = nn.CrossEntropyLoss()

    print(model)

    loss_mean_train = np.zeros(config["number_epoch"])
    loss_std_train = np.zeros(config["number_epoch"])
    accuracy_mean_train = np.zeros(config["number_epoch"])
    accuracy_std_train = np.zeros(config["number_epoch"])

    loss_mean_test = np.zeros(config["number_epoch"])
    loss_std_test = np.zeros(config["number_epoch"])
    accuracy_mean_test = np.zeros(config["number_epoch"])
    accuracy_std_test = np.zeros(config["number_epoch"])

    log_path = os.path.join(config["result_path"], "log.txt")

    # training
    for i in range(config["number_epoch"]):
        loss_train_epoch = []
        accuracy_train_epoch = []

        model.train()

        for _, (images, labels) in enumerate(tqdm(config["dataloader_train"])):
            images = images.to(config["device"])
            labels = labels.to(config["device"])

            prediction_train = model(images)
            loss_train = criterion(prediction_train, labels)
            accuracy_train = compute_accuracy(prediction_train, labels)

            back_optimizer.zero_grad()
            front_optimizer.zero_grad()

            loss_train.backward(retain_graph=True)

            back_optimizer.step()
            front_optimizer.step()

            loss_train_epoch.append(loss_train.item())
            accuracy_train_epoch.append(accuracy_train)

        loss_mean_train[i] = np.mean(loss_train_epoch)
        loss_std_train[i] = np.std(loss_train_epoch)
        accuracy_mean_train[i] = np.mean(accuracy_train_epoch)
        accuracy_std_train[i] = np.std(accuracy_train_epoch)

        # testing
        loss_test_epoch = []
        accuracy_test_epoch = []

        model.eval()

        with torch.no_grad():
            for _, (images, labels) in enumerate(config["dataloader_test"]):
                images = images.to(config["device"])
                labels = labels.to(config["device"])

                prediction_test = model(images)
                loss_test = criterion(prediction_test, labels)
                accuracy_test = compute_accuracy(prediction_test, labels)

                loss_test_epoch.append(loss_test.item())
                accuracy_test_epoch.append(accuracy_test)

        loss_mean_test[i] = np.mean(loss_test_epoch)
        loss_std_test[i] = np.std(loss_test_epoch)
        accuracy_mean_test[i] = np.mean(accuracy_test_epoch)
        accuracy_std_test[i] = np.std(accuracy_test_epoch)

        # print results
        log_train = "%3d, : [train loss : %.5f , train accuracy : %.5f ]" % (
            i, loss_mean_train[i], accuracy_mean_train[i])
        log_test = "%3d, : [test loss : %.5f , test accuracy : %.5f ]" % (
            i, loss_mean_test[i], accuracy_mean_test[i])
        print(log_train)
        print(log_test)
        print('########################################################')

        # write log file
        log = "%3d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f" \
            % (i, loss_mean_train[i], loss_std_train[i], accuracy_mean_train[i], accuracy_std_train[i],
               loss_mean_test[i], loss_std_test[i], accuracy_mean_test[i], accuracy_std_test[i])
        with open(log_path, "a") as f:
            f.write(log + "\n")

        # plot results
        plot_results_per_epoch(loss_mean_train, loss_std_train,
                               "iteration", "loss", "train loss", config["result_path"])
        plot_results_per_epoch(loss_mean_test, loss_std_test,
                               "iteration", "loss", "test loss", config["result_path"])
        plot_results_per_epoch(accuracy_mean_train, accuracy_std_train,
                               "iteration", "loss", "train accuracy", config["result_path"])
        plot_results_per_epoch(accuracy_mean_test, accuracy_std_test,
                               "iteration", "loss", "test accuracy", config["result_path"])

    # save model
    model_path = os.path.join(config["result_path"], "model.pth")
    torch.save(model.state_dict(), model_path)


def train_proximal_classifier(config):
    if config["model"] == "linear_classifier":
        model = LinearClassifier(
            img_size=config["img_size"], img_channel=config["img_channel"]).to(config["device"])
    elif config["model"] == "basic_classifier":
        model = BasicClassifier(
            config["img_size"], config["img_channel"], config["bottleneck_pos"]).to(config["device"])
    else:
        raise NotImplementedError

    if config["optimizer"] == "adam":
        front_optimizer = torch.optim.Adam(
            model.front_network.parameters(), lr=config["learning_rate"][0])
        back_optimizer = torch.optim.Adam(
            model.back_network.parameters(), lr=config["learning_rate"][1])
        # z_optimizer = torch.optim.Adam(
        #     [z], lr=config["learning_rate"][2])
    elif config["optimizer"] == "adadelta":
        front_optimizer = torch.optim.Adadelta(model.front_network.parameters(
        ), lr=config["learning_rate"][0], rho=0.9, eps=1e-3, weight_decay=0.001)
        back_optimizer = torch.optim.Adadelta(model.back_network.parameters(
        ), lr=config["learning_rate"][1], rho=0.9, eps=1e-3, weight_decay=0.001)
        # z_optimizer = torch.optim.Adadelta([z], lr=config["learning_rate"][2], rho=0.9, eps=1e-3, weight_decay=0.001)
    elif config["optimizer"] == "sgd":
        front_optimizer = torch.optim.SGD(model.front_network.parameters(
        ), lr=config["learning_rate"][0])
        back_optimizer = torch.optim.SGD(model.back_network.parameters(
        ), lr=config["learning_rate"][1])
        # z_optimizer = torch.optim.SGD([z], lr=config["learning_rate"][2])
    else:
        raise NotImplementedError

    if config["front_loss"] == "l1":
        front_criterion = nn.L1Loss()
    elif config["front_loss"] == "l2":
        front_criterion = nn.MSELoss()
    else:
        raise NotImplementedError

    back_criterion = nn.CrossEntropyLoss()

    print(model)

    loss_mean_train = np.zeros(config["number_epoch"])
    loss_std_train = np.zeros(config["number_epoch"])
    accuracy_mean_train = np.zeros(config["number_epoch"])
    accuracy_std_train = np.zeros(config["number_epoch"])

    loss_mean_test = np.zeros(config["number_epoch"])
    loss_std_test = np.zeros(config["number_epoch"])
    accuracy_mean_test = np.zeros(config["number_epoch"])
    accuracy_std_test = np.zeros(config["number_epoch"])

    log_path = os.path.join(config["result_path"], "log.txt")

    # training
    for i in range(config["number_epoch"]):
        loss_train_epoch = []
        accuracy_train_epoch = []

        model.train()

        for _, (images, labels, z_idxs) in enumerate(tqdm(config["dataloader_train"])):
            images = images.to(config["device"])
            labels = labels.to(config["device"])
            z_idxs = z_idxs.to(config["device"])

            # update back parameters
            z = model.front_network(images)
            z_var = nn.Parameter(z, requires_grad=True)
            back_optimizer.zero_grad()
            prediction_train = model.back_network(z_var)
            loss = back_criterion(prediction_train, labels)
            loss.backward(retain_graph=True)
            back_optimizer.step()

            # update z
            z_var_init = z_var.data
            for _ in range(config["learning_rate"][-1]):
                if z_var.grad is not None:
                    z_var.grad.zero_()
                prediction_train = model.back_network(z_var)
                loss_back = back_criterion(prediction_train, labels)
                loss_front = front_criterion(z_var, z_var_init.detach())
                loss = loss_back + config["learning_rate"][3] / 2. * loss_front
                loss.backward(retain_graph=True)
                z_var.data = z_var - config["learning_rate"][2] * z_var.grad

            z_update_value = z_var.data

            # update front parameters
            front_optimizer.zero_grad()
            loss = config["learning_rate"][3] / 2. * \
                front_criterion(z_update_value.detach(), z)
            loss.backward(retain_graph=True)
            front_optimizer.step()

            # calculate train loss
            with torch.no_grad():
                prediction_train = model(images)
                loss_train = back_criterion(prediction_train, labels)

            accuracy_train = compute_accuracy(prediction_train, labels)

            loss_train_epoch.append(loss_train.item())
            accuracy_train_epoch.append(accuracy_train)

        loss_mean_train[i] = np.mean(loss_train_epoch)
        loss_std_train[i] = np.std(loss_train_epoch)
        accuracy_mean_train[i] = np.mean(accuracy_train_epoch)
        accuracy_std_train[i] = np.std(accuracy_train_epoch)

        # testing
        loss_test_epoch = []
        accuracy_test_epoch = []

        model.eval()

        with torch.no_grad():
            for _, (images, labels) in enumerate(config["dataloader_test"]):
                images = images.to(config["device"])
                labels = labels.to(config["device"])

                prediction_test = model(images)
                loss_test = back_criterion(prediction_test, labels)
                accuracy_test = compute_accuracy(prediction_test, labels)

                loss_test_epoch.append(loss_test.item())
                accuracy_test_epoch.append(accuracy_test)

        loss_mean_test[i] = np.mean(loss_test_epoch)
        loss_std_test[i] = np.std(loss_test_epoch)
        accuracy_mean_test[i] = np.mean(accuracy_test_epoch)
        accuracy_std_test[i] = np.std(accuracy_test_epoch)

        # print results
        log_train = "%3d, : [train loss : %.10f , train accuracy : %.5f ]" \
            % (i, loss_mean_train[i],  accuracy_mean_train[i])
        log_test = "%3d, : [test loss : %.5f , test accuracy : %.5f ]" \
            % (i, loss_mean_test[i], accuracy_mean_test[i])
        print(log_train)
        print(log_test)
        print('########################################################')

        # write log file
        log = "%3d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f" \
            % (i, loss_mean_train[i], loss_std_train[i],
               accuracy_mean_train[i], accuracy_std_train[i], loss_mean_test[i], loss_std_test[i], accuracy_mean_test[i], accuracy_std_test[i])
        with open(log_path, "a") as f:
            f.write(log + "\n")

        # plot results
        plot_results_per_epoch(loss_mean_train, loss_std_train,
                               "iteration", "loss", "train loss", config["result_path"])
        plot_results_per_epoch(loss_mean_test, loss_std_test,
                               "iteration", "loss", "test loss", config["result_path"])
        plot_results_per_epoch(accuracy_mean_train, accuracy_std_train,
                               "iteration", "loss", "train accuracy", config["result_path"])
        plot_results_per_epoch(accuracy_mean_test, accuracy_std_test,
                               "iteration", "loss", "test accuracy", config["result_path"])

    # save model
    model_path = os.path.join(config["result_path"], "model.pth")
    torch.save(model.state_dict(), model_path)


def train_admm_classifier(config):
    if config["model"] == "linear_classifier":
        model = LinearClassifier(
            img_size=config["img_size"], img_channel=config["img_channel"]).to(config["device"])
    elif config["model"] == "basic_classifier":
        model = BasicClassifier(
            config["img_size"], config["img_channel"], config["bottleneck_pos"]).to(config["device"])
    else:
        raise NotImplementedError

    y = torch.nn.Parameter(torch.rand(
        (len(config["dataloader_train"].dataset), model.z_dim))).to(config["device"])
    y = torch.nn.Parameter(y, requires_grad=False)

    if config["optimizer"] == "adam":
        front_optimizer = torch.optim.Adam(
            model.front_network.parameters(), lr=config["learning_rate"][0])
        back_optimizer = torch.optim.Adam(
            model.back_network.parameters(), lr=config["learning_rate"][1])
        # z_optimizer = torch.optim.Adam(
        #     [z], lr=config["learning_rate"][2])
        # y_optimizer = torch.optim.Adam(
        #     [y], lr=config["learning_rate"][3])
    elif config["optimizer"] == "adadelta":
        front_optimizer = torch.optim.Adadelta(model.front_network.parameters(
        ), lr=config["learning_rate"][0], rho=0.9, eps=1e-3, weight_decay=0.001)
        back_optimizer = torch.optim.Adadelta(model.back_network.parameters(
        ), lr=config["learning_rate"][1], rho=0.9, eps=1e-3, weight_decay=0.001)
        # z_optimizer = torch.optim.Adadelta([z], lr=config["learning_rate"][2], rho=0.9, eps=1e-3, weight_decay=0.001)
        # y_optimizer = torch.optim.Adadelta([y], lr=config["learning_rate"][3], rho=0.9, eps=1e-3, weight_decay=0.001)
    elif config["optimizer"] == "sgd":
        front_optimizer = torch.optim.SGD(model.front_network.parameters(
        ), lr=config["learning_rate"][0])
        back_optimizer = torch.optim.SGD(model.back_network.parameters(
        ), lr=config["learning_rate"][1])
        # z_optimizer = torch.optim.SGD([z], lr=config["learning_rate"][2])
        # y_optimizer = torch.optim.SGD([y], lr=config["learning_rate"][3])
    else:
        raise NotImplementedError

    if config["front_loss"] == "l1":
        front_criterion = nn.L1Loss()
    elif config["front_loss"] == "l2":
        front_criterion = nn.MSELoss()
    else:
        raise NotImplementedError

    back_criterion = nn.CrossEntropyLoss()

    print(model)

    loss_mean_train = np.zeros(config["number_epoch"])
    loss_std_train = np.zeros(config["number_epoch"])
    accuracy_mean_train = np.zeros(config["number_epoch"])
    accuracy_std_train = np.zeros(config["number_epoch"])

    loss_mean_test = np.zeros(config["number_epoch"])
    loss_std_test = np.zeros(config["number_epoch"])
    accuracy_mean_test = np.zeros(config["number_epoch"])
    accuracy_std_test = np.zeros(config["number_epoch"])

    log_path = os.path.join(config["result_path"], "log.txt")

    # training
    for i in range(config["number_epoch"]):
        loss_train_epoch = []
        accuracy_train_epoch = []

        model.train()

        for _, (images, labels, z_idxs) in enumerate(tqdm(config["dataloader_train"])):
            images = images.to(config["device"])
            labels = labels.to(config["device"])
            z_idxs = z_idxs.to(config["device"])

            # update back parameters
            z = model.front_network(images)
            z_var = nn.Parameter(z, requires_grad=True)
            back_optimizer.zero_grad()
            prediction_train = model.back_network(z_var)
            loss = back_criterion(prediction_train, labels)
            loss.backward(retain_graph=True)
            back_optimizer.step()

            # update z
            z_var_init = z_var.data
            for _ in range(config["learning_rate"][-1]):
                if z_var.grad is not None:
                    z_var.grad.zero_()
                prediction_train = model.back_network(z_var)
                loss_back = back_criterion(prediction_train, labels)
                loss_front = front_criterion(
                    z_var + y[z_idxs].detach(), z_var_init.detach())
                loss = loss_back + config["learning_rate"][4] / 2. * loss_front
                loss.backward(retain_graph=True)
                z_var.data = z_var - config["learning_rate"][2] * z_var.grad

            z_update_value = z_var.data

            # update y
            y[z_idxs].data = y[z_idxs] + (z_update_value.detach() - z.detach())

            # update front parameters
            front_optimizer.zero_grad()
            loss = config["learning_rate"][4] / 2. * \
                front_criterion(z_update_value.detach() +
                                y[z_idxs].detach(), z)
            loss.backward(retain_graph=True)
            front_optimizer.step()

            # update y
            # y[z_idxs].data = y[z_idxs] + (z_update_value.detach() - model.front_network(images).detach())

            # calculate train loss
            with torch.no_grad():
                prediction_train = model(images)
                loss_train = back_criterion(prediction_train, labels)

            accuracy_train = compute_accuracy(prediction_train, labels)

            loss_train_epoch.append(loss_train.item())
            accuracy_train_epoch.append(accuracy_train)

        loss_mean_train[i] = np.mean(loss_train_epoch)
        loss_std_train[i] = np.std(loss_train_epoch)
        accuracy_mean_train[i] = np.mean(accuracy_train_epoch)
        accuracy_std_train[i] = np.std(accuracy_train_epoch)

        # testing
        loss_test_epoch = []
        accuracy_test_epoch = []

        model.eval()

        with torch.no_grad():
            for _, (images, labels) in enumerate(config["dataloader_test"]):
                images = images.to(config["device"])
                labels = labels.to(config["device"])

                prediction_test = model(images)
                loss_test = back_criterion(prediction_test, labels)
                accuracy_test = compute_accuracy(prediction_test, labels)

                loss_test_epoch.append(loss_test.item())
                accuracy_test_epoch.append(accuracy_test)

        loss_mean_test[i] = np.mean(loss_test_epoch)
        loss_std_test[i] = np.std(loss_test_epoch)
        accuracy_mean_test[i] = np.mean(accuracy_test_epoch)
        accuracy_std_test[i] = np.std(accuracy_test_epoch)

        # print results
        log_train = "%3d, : [train loss : %.10f , train accuracy : %.5f ]" \
            % (i, loss_mean_train[i],  accuracy_mean_train[i])
        log_test = "%3d, : [test loss : %.5f , test accuracy : %.5f ]" \
            % (i, loss_mean_test[i], accuracy_mean_test[i])
        print(log_train)
        print(log_test)
        print('########################################################')

        # write log file
        log = "%3d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f" \
            % (i, loss_mean_train[i], loss_std_train[i],
               accuracy_mean_train[i], accuracy_std_train[i], loss_mean_test[i], loss_std_test[i], accuracy_mean_test[i], accuracy_std_test[i])
        with open(log_path, "a") as f:
            f.write(log + "\n")

        # plot results
        plot_results_per_epoch(loss_mean_train, loss_std_train,
                               "iteration", "loss", "train loss", config["result_path"])
        plot_results_per_epoch(loss_mean_test, loss_std_test,
                               "iteration", "loss", "test loss", config["result_path"])
        plot_results_per_epoch(accuracy_mean_train, accuracy_std_train,
                               "iteration", "loss", "train accuracy", config["result_path"])
        plot_results_per_epoch(accuracy_mean_test, accuracy_std_test,
                               "iteration", "loss", "test accuracy", config["result_path"])

    # save model
    model_path = os.path.join(config["result_path"], "model.pth")
    torch.save(model.state_dict(), model_path)
