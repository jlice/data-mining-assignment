#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from data import *
from layer import *
from util import *


def pipeline(features, target, layers, loss, **params):
    """训练工作流，执行训练流程

    Args:
        features: 特征
        target: 目标
        layers: 网络层结构
        loss: 损失定义
        **params:
            batch_size: 批大小
            learning_rate: 学习率（数字或以步数为自变量的函数）
            train_ratio: 训练集所占的比例
            seed: 随机化种子
            step: 训练步数
            print_step: 每多少步输出一次信息

    Returns: None

    """
    batch_size = params.get('batch_size', 128)
    learning_rate = params.get('learning_rate', 1)
    train_ratio = params.get('train_ratio', 0.8)
    seed = params.get('seed', 1024)
    train_step = params.get('step')
    print_step = params.get('print_step', 10)

    train_x, train_y, test_x, test_y = train_test_split(features, target, train_ratio, seed)
    batch = data_iter(train_x, train_y, batch_size, seed)
    if isinstance(learning_rate, (int, float)):
        for layer in layers:
            layer.lr = learning_rate
    step = 1
    epoch = 0
    loss_sum = 0
    need_stop = False
    print('Start Training!')
    while True:
        if need_stop:
            break

        epoch += 1
        for _ in range(0, len(train_y), batch_size):
            if isinstance(learning_rate, type(lambda: None)):
                for layer in layers:
                    layer.lr = learning_rate(step)

            batch_x, batch_y = next(batch)
            x = [row[:] for row in batch_x]
            for layer in layers:
                x = layer.forward(x)
            loss_sum += loss.forward(x, batch_y)
            delta = loss.backward()
            for layer in layers[::-1]:
                delta = layer.backward(delta)

            if step % print_step == 0:
                x = [row[:] for row in test_x]
                for layer in layers:
                    x = layer.forward(x)
                accu = accuracy(x, test_y)
                test_loss = loss.forward(x, test_y)
                print('epoch:{:<4}  step:{:<6} train_loss:{:.5f}   test_loss:{:.5f}   accuracy:{:.5f}'.format(
                    epoch, step, loss_sum / print_step, test_loss, accu))
                loss_sum = 0

            step += 1
            if train_step is not None and step > train_step:
                need_stop = True
                break


def iris():
    """训练iris数据集

    Returns: None

    """
    features, target = load_csv('iris/iris.csv', features_type=float)
    layers = [
        Dense(4, 6),
        Sigmoid(),
        Dense(6, 3),
        Sigmoid()
    ]
    loss = L2Loss()
    params = {'step': 400, 'batch_size': 32, 'learning_rate': 7, 'train_ratio': 0.8, 'seed': 1024}
    pipeline(features, target, layers, loss, **params)


def mnist():
    """训练MNIST数据集

    Returns: None

    """
    features, target = load_csv('MNIST-data/mnist.csv', features_type=int)
    features = each(features, lambda x: x / 255)
    layers = [
        Dense(28 * 28, 10),
        Sigmoid()
    ]
    loss = L2Loss()
    params = {'step': 5000, 'print_step': 10, 'batch_size': 32, 'train_ratio': 0.85, 'seed': 1024,
              'learning_rate': lambda x: 5 / 1.1 ** (x // 100)}
    pipeline(features, target, layers, loss, **params)


if __name__ == '__main__':
    iris()
    mnist()
