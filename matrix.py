#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def shape(x):
    """计算多维数组的形状
    
    Args:
        x: 数组

    Returns: 数组的形状
    
    Examples:
        >>> shape(1)
        ()
        >>> shape([1, 2])
        (2,)
        >>> shape([[1, 1], [2, 2], [3, 3]])
        (3, 2)
        >>> shape([[[1, 2, 3], [4, 5, 6]]])
        (1, 2, 3)

    """
    if isinstance(x, (list, tuple)):
        value = []
        tmp = x
        while isinstance(tmp, (list, tuple)):
            value.append(len(tmp))
            tmp = tmp[0]
        return tuple(value)
    else:
        return tuple()


def reshape(x, row, col):
    """将数组变形为矩阵
    
    Args:
        x: 数组
        row: 行数
        col: 列数

    Returns: 矩阵
    
    Examples:
        >>> reshape([1, 2, 3, 4, 5, 6], 2, 3)
        [[1, 2, 3], [4, 5, 6]]

    """
    if len(x) != row * col:
        raise ValueError("不能将长度为%d的数组变形为%dx%d大小的矩阵" % (len(x), row, col))
    reshaped = []
    for i in range(row):
        reshaped.append([])
        for j in range(col):
            reshaped[i].append(x[col * i + j])
    return reshaped


def each(x, func):
    """对数组或矩阵中每个元素执行计算
    
    Args:
        x: 数组或矩阵
        func: 要执行的计算

    Returns: 执行计算后的数组或矩阵
    
    Examples:
        >>> each([1, 2, 3], lambda x: x * 2)
        [2, 4, 6]
        >>> each([[1, 2], [3, 4]], lambda x: x * 2)
        [[2, 4], [6, 8]]

    """
    if len(shape(x)) == 1:
        return list(map(func, x))
    elif len(shape(x)) == 2:
        m, n = shape(x)
        result = reshape([None] * (m * n), m, n)
        for i in range(m):
            for j in range(n):
                result[i][j] = func(x[i][j])
        return result
    else:
        raise ValueError("输入错误：期望是数组或矩阵")


def accord(a, b, op):
    """形状相同的两个数组或矩阵对应位置元素执行计算
    
    Args:
        a: 数组或矩阵
        b: 数组或矩阵
        op: 要执行的运算：四则运算或函数

    Returns: 执行计算后的数组或矩阵
    
    Examples:
        >>> accord([[1, 2]], [[3, 4]], '+')
        [[4, 6]]
        >>> accord([5, 3], [2, 3], '-')
        [3, 0]
        >>> accord([[2, 4]], [[1, 3]], lambda x, y: 2 * x + y)
        [[5, 11]]

    """
    if op in ('+', '-', '*', '/'):
        op = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
        }[op]
    elif not isinstance(op, type(lambda: None)):
        raise ValueError("输入的运算不合法")

    if len(shape(a)) == 1 and len(shape(b)) == 1:
        return [op(ai, bi) for ai, bi in zip(a, b)]
    elif len(shape(a)) == 2 and len(shape(b)) == 2:
        m, n = shape(a)
        result = reshape([None] * (m * n), m, n)
        for i in range(m):
            for j in range(n):
                result[i][j] = op(a[i][j], b[i][j])
        return result
    else:
        raise ValueError("输入错误：期望是两个数组或矩阵且二者形状相同")


def sum_axis(x, axis=0):
    """对矩阵某个轴进行求和
    
    Args:
        x: 矩阵
        axis: 轴序号

    Returns: 求和的结果
    
    Examples:
        >>> sum_axis([[1, 2, 3], [3, 4, 5]], axis=0)
        [4, 6, 8]
        >>> sum_axis([[1, 2, 3], [3, 4, 5]], axis=1)
        [6, 12]

    """
    if len(shape(x)) == 2:
        m, n = shape(x)
        if axis == 0:
            result = [None] * n
            for i in range(n):
                result[i] = sum([xi[i] for xi in x])
            return result
        elif axis == 1:
            result = [None] * m
            for i in range(m):
                result[i] = sum([xi for xi in x[i]])
            return result
    else:
        raise ValueError("输入错误：期望是一个矩阵")


def transpose(x):
    """转置矩阵
    
    Args:
        x: 矩阵

    Returns: 转置后的矩阵
    
    Examples:
        >>> transpose([[1, 2, 3]])
        [[1], [2], [3]]

    """
    if len(shape(x)) != 2:
        raise ValueError("输入错误：期望是一个矩阵")
    m, n = shape(x)
    result = reshape([None] * (m * n), n, m)
    for i in range(m):
        for j in range(n):
            result[j][i] = x[i][j]
    return result


def dot(a, b):
    """矩阵点乘

    Args:
        a: 矩阵
        b: 矩阵

    Returns: 矩阵点乘

    Examples:
        >>> dot([[1, 2]], [[3], [4]])
        [[11]]
        >>> dot([[1], [2]], [[3, 4]])
        [[3, 4], [6, 8]]

    """
    if len(shape(a)) != 2 or len(shape(b)) != 2:
        raise ValueError("输入错误：期望是两个矩阵")
    if shape(a)[1] != shape(b)[0]:
        raise ValueError("形状不匹配：%s与%s，不能执行点乘" % (shape(a), shape(b)))
    m, n = shape(a)
    s = shape(b)[1]
    result = reshape([None] * (m * s), m, s)
    for i in range(m):
        for j in range(s):
            tmp = 0
            for k in range(n):
                tmp += a[i][k] * b[k][j]
            result[i][j] = tmp
    return result
