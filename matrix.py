#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def shape(x):
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
    """reshape a list or tuple to a matrix"""
    if len(x) != row * col:
        raise ValueError("Can't reshape to (%d, %d) from length %d" % (row, col, len(x)))
    reshaped = []
    for i in range(row):
        reshaped.append([])
        for j in range(col):
            reshaped[i].append(x[col * i + j])
    return reshaped


def each(x, func):
    """
    x: matrix(mxn)
    逐元素计算

    >>> each([[1, 2], [3, 4]], lambda x: x ** 2)
    [[1, 4], [9, 16]]
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
        raise ValueError("Input Error: expect 2-dim list")


def accord(a, b, op):
    """
    a, b: matrix(mxn)
    a, b: list(n)
    >>> accord([[1, 2]], [[3, 4]], '+')
    [[4, 6]]
    >>> accord([5, 3], [2, 3], '-')
    [3, 0]
    """
    if op in ('+', '-', '*', '/'):
        op = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
        }[op]
    elif not isinstance(op, type(lambda: None)):
        raise ValueError("输入的操作不合法")

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
        raise ValueError("输入数组维数不合法")


def sum_axis(x, axis=0):
    """
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
        raise ValueError("输入数组维数不合法")


def transpose(x):
    """
    x: matrix(mxn)

    >>> transpose([[1, 2, 3]])
    [[1], [2], [3]]
    """
    if len(shape(x)) != 2:
        raise ValueError("Input Error: expect 2-dim list")
    m, n = shape(x)
    result = reshape([None] * (m * n), n, m)
    for i in range(m):
        for j in range(n):
            result[j][i] = x[i][j]
    return result


def dot(a, b):
    """
    a: matrix(mxn)
    b: matrix(nxs)
    return: mxs
    """
    if len(shape(a)) != 2 or len(shape(b)) != 2:
        raise ValueError("Input Error: expect two matrices")
    if shape(a)[1] != shape(b)[0]:
        raise ValueError("Shapes dismatch for %s and %s" % (shape(a), shape(b)))
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
