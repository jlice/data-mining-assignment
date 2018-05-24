#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matrix import transpose, each
from rand import shuffle


def data_iter(features, target, batch_size=32, seed=1024):
    num = len(target)
    idx = list(range(num))
    while True:
        shuffle(idx, seed)
        seed *= 2
        for i in range(0, num, batch_size):
            part = idx[i: min(i + batch_size, num)]
            yield [features[j] for j in part], [target[j] for j in part]


def train_test_split(features, target, train_ratio, seed=65536):
    shuffle(features, seed)
    shuffle(target, seed)
    train_x = features[:int(len(target) * train_ratio)]
    train_y = target[:int(len(target) * train_ratio)]
    test_x = features[int(len(target) * train_ratio):]
    test_y = target[int(len(target) * train_ratio):]
    return train_x, train_y, test_x, test_y


def min_max_scaler(features):
    """
    归一化

    >>> min_max_scaler([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
    [[0.0, 0.0], [0.25, 0.25], [0.5, 0.5], [1.0, 1.0]]
    """
    transposed = transpose(features)
    for i in range(len(transposed)):
        min_value, max_value = min(transposed[i]), max(transposed[i])
        transposed[i] = each(transposed[i], lambda x: (x - min_value) / (max_value - min_value))
    return transpose(transposed)


def accuracy(pred, target):
    counter = 0
    for i in range(len(pred)):
        max_pred = 0
        max_index = 0
        for j in range(len(pred[0])):
            if pred[i][j] >= max_pred:
                max_index = j
                max_pred = pred[i][j]
        if max_index == target[i]:
            counter += 1
    return counter / len(pred)
