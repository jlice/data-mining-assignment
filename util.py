#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matrix import transpose, each
from rand import shuffle


def data_iter(features, target, batch_size=32, seed=1024):
    """数据集批次生成器。每次从数据集中返回一个批次，每轮迭代前会打乱数据集

    Args:
        features: 特征
        target: 目标
        batch_size: 批大小
        seed: 随机化种子

    Returns: 生成器

    """
    num = len(target)
    idx = list(range(num))
    while True:
        shuffle(idx, seed)
        seed *= 2
        for i in range(0, num, batch_size):
            part = idx[i: min(i + batch_size, num)]
            yield [features[j] for j in part], [target[j] for j in part]


def train_test_split(features, target, train_ratio, seed=65536):
    """数据集划分为训练集和测试集

    Args:
        features: 特征
        target: 目标
        train_ratio: 训练集所占的比例
        seed: 随机化种子

    Returns: 训练集特征，训练集目标，测试集特征，测试集目标

    """
    shuffle(features, seed)
    shuffle(target, seed)
    train_x = features[:int(len(target) * train_ratio)]
    train_y = target[:int(len(target) * train_ratio)]
    test_x = features[int(len(target) * train_ratio):]
    test_y = target[int(len(target) * train_ratio):]
    return train_x, train_y, test_x, test_y


def min_max_scaler(features):
    """对每个特征按最大最小值进行线性缩放归一化至[0, 1]
    x = (x - min) / (max - min)

    Args:
        features: 特征

    Returns: 归一化后的特征

    Examples:
        >>> min_max_scaler([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
        [[0.0, 0.0], [0.25, 0.25], [0.5, 0.5], [1.0, 1.0]]

    """
    transposed = transpose(features)
    for i in range(len(transposed)):
        min_value, max_value = min(transposed[i]), max(transposed[i])
        transposed[i] = each(transposed[i], lambda x: (x - min_value) / (max_value - min_value))
    return transpose(transposed)


def accuracy(pred, target):
    """计算准确率

    Args:
        pred: 预测值（每个分类的确信度）
        target: 目标（是哪个分类）

    Returns: 准确率

    """
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
