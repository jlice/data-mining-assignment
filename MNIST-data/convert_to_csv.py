#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gzip
import struct

import numpy as np


def load_images(image_gz):
    with gzip.open(image_gz) as f:
        buf = f.read()
    num = int(struct.unpack_from('>i', buf, 4)[0])
    return np.array(struct.unpack_from('B' * num * 28 * 28, buf, 16)).reshape(num, 784)


def load_labels(label_gz):
    with gzip.open(label_gz) as f:
        buf = f.read()
    num = int(struct.unpack_from('>i', buf, 4)[0])
    return np.array(struct.unpack_from('B' * num, buf, 8))


def write_csv(mnist_csv):
    train_images = load_images('train-images-idx3-ubyte.gz')
    train_labels = load_labels('train-labels-idx1-ubyte.gz')
    test_images = load_images('t10k-images-idx3-ubyte.gz')
    test_labels = load_labels('t10k-labels-idx1-ubyte.gz')
    with open(mnist_csv, 'w') as f:
        for i in range(train_labels.shape[0]):
            f.write('%s,%d\n' % (','.join(map(str, train_images[i])), train_labels[i]))
        for i in range(test_labels.shape[0]):
            f.write('%s,%d\n' % (','.join(map(str, test_images[i])), test_labels[i]))


if __name__ == '__main__':
    write_csv('mnist.csv')
