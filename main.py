#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from data import *
from layer import *
from util import *


def pipeline(features, target, layers, loss, **params):
    batch_size = params.get('batch_size', 128)
    learning_rate = params.get('learning_rate', 0.1)
    train_ratio = params.get('train_ratio', 0.8)
    seed = params.get('seed', 1024)
    train_step = params.get('step')
    print_step = params.get('print_step', 10)

    train_x, train_y, test_x, test_y = train_test_split(features, target, train_ratio, seed)
    batch = data_iter(train_x, train_y, batch_size, seed)
    for layer in layers:
        layer.lr = learning_rate
    step = 1
    epoch = 0
    need_stop = False
    print('Start Training!')
    while True:
        if need_stop:
            break
        epoch += 1
        loss_sum = 0
        for _ in range(0, len(train_y), batch_size):
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
                print('epoch:{:<4}  step:{:<6} loss:{:.5f}   accuracy:{:.5f}'.format(
                    epoch, step, loss_sum / print_step, accu))

            step += 1
            if train_step is not None and step >= train_step:
                need_stop = True
                break


def iris():
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


if __name__ == '__main__':
    iris()
