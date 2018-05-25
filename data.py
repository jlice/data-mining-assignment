#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def load_csv(csv, features_type=float, target_type=int):
    """加载csv。最后一列为目标列，其余为特征。

    Args:
        csv: csv文件名
        features_type: 特征的数据类型
        target_type: 目标的数据类型

    Returns: 特征, 目标

    """
    features = []
    target = []
    for i, line in enumerate(open(csv)):
        if line.strip():
            data = line.strip().split(',')
            features.append(list(map(features_type, data[:-1])))
            target.append(target_type(data[-1]))
    return features, target
