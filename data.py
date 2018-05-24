#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def load_csv(csv, features_type, target_type=int):
    features = []
    target = []
    for i, line in enumerate(open(csv)):
        if line.strip():
            data = line.strip().split(',')
            features.append(list(map(features_type, data[:-1])))
            target.append(target_type(data[-1]))
    return features, target
