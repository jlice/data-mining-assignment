#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def rand(n, seed=65536):
    x = seed
    result = []
    for _ in range(n):
        x = (x * 1103515245 + 12345) % (2 ** 32)
        result.append(x / 2 ** 32)
    return result


def rand_normal(n, seed=65536):
    source = rand(n * 192, seed)
    result = []
    for i in range(n):
        x = (sum(source[192 * i:192 * (i + 1)]) - 96) / 4
        result.append(x)
    return result


def shuffle(x, seed=65536):
    source = rand(len(x) * 192, seed)
    for i in reversed(range(1, len(x))):
        j = int(source[i] * (i + 1))
        x[i], x[j] = x[j], x[i]
