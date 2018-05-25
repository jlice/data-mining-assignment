#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matrix import *
from rand import rand_normal


class Dense:
    """全连接层

    Args:
        input_num: 输入节点数
        units: 输出节点数

    Attributes:
        inputs: 输入
        inputs_grad: 输入的梯度
        units: 输出节点数
        kernel: 权值
        kernel_grad: 权值的梯度
        bias: 偏置
        bias_grad: 偏置的梯度
        lr: 学习率
        y: 输出

    """
    def __init__(self, input_num, units):
        self.inputs = None
        self.inputs_grad = None
        self.units = units
        self.kernel = reshape([x / 10 for x in rand_normal(input_num * units)], input_num, units)
        self.kernel_grad = [[0] * units for _ in range(input_num)]
        self.bias = [0] * units
        self.bias_grad = [0] * units
        self.lr = 0
        self.y = None

    def forward(self, inputs):
        """前向传播

        Args:
            inputs: 输入

        Returns: 前向传播时层的输出

        """
        self.inputs = inputs
        self.inputs_grad = inputs
        self.y = accord(dot(inputs, self.kernel), [self.bias] * len(inputs), '+')
        return self.y

    def backward(self, delta):
        """反向传播，并执行参数更新

        Args:
            delta: 误差

        Returns: 输入的梯度

        """
        self.bias_grad = each(sum_axis(delta, axis=0), lambda x: x / len(delta))
        self.kernel_grad = each(dot(transpose(self.inputs), delta), lambda x: x / len(delta))

        self.inputs_grad = dot(delta, transpose(self.kernel))
        self.kernel = accord(self.kernel, each(self.kernel_grad, lambda x: -1 * x * self.lr), '+')
        self.bias = accord(self.bias, each(self.bias_grad, lambda x: -1 * x * self.lr), '+')

        return self.inputs_grad


class Sigmoid:
    """Sigmoid激活层

    Attributes:
        inputs: 输入
        inputs_grad: 输入的梯度
        y: 输出

    """
    E = 2.71828182846

    def __init__(self):
        self.inputs = None
        self.inputs_grad = None
        self.y = []

    @staticmethod
    def _sigmoid(inputs):
        """sigmoid函数
        f(x) = 1 / (1 + exp(-x))

        Args:
            inputs: 输入自变量

        Returns: sigmoid计算的结果

        """
        return each(inputs, lambda x: 1 / (1 + __class__.E ** (-1 * x)))

    def forward(self, inputs):
        """前向传播

        Args:
            inputs: 输入

        Returns: 输出

        """
        self.inputs = inputs
        self.inputs_grad = inputs
        self.y = __class__._sigmoid(inputs)
        return self.y

    def backward(self, delta):
        """反向传播

        Args:
            delta: 误差

        Returns: 输入的梯度

        """
        sig = __class__._sigmoid(self.inputs)
        self.inputs_grad = accord(delta, accord(sig, each(sig, lambda x: 1 - x), '*'), '*')
        return self.inputs_grad


class L2Loss:
    """平方损失层

        Attributes:
            pred: 预测值（每个分类的确信度）
            pred_grad: 预测值的梯度
            label: 标签（标记的每个分类的确信度）
            loss: 损失值

        """
    def __init__(self):
        self.pred = None
        self.pred_grad = None
        self.label = None
        self.loss = 0

    def forward(self, pred, target):
        """前向传播

        Args:
            pred: 预测值
            target: 目标（是哪个分类）

        Returns: 损失值

        """
        self.pred = pred
        self.pred_grad = pred
        self.label = reshape([0] * len(pred) * len(pred[0]), len(pred), len(pred[0]))
        for a, b in zip(self.label, target):
            a[b] = 1
        self.loss = sum(sum_axis(each(accord(pred, self.label, '-'), lambda x: x ** 2), axis=0)) / (2 * len(pred))
        return self.loss

    def backward(self):
        """反向传播

        Returns: 预测值的梯度

        """
        self.pred_grad = each(accord(self.pred, self.label, '-'), lambda x: x / len(self.pred))
        return self.pred_grad
