#!/bin/bash

URL=http://yann.lecun.com/exdb/mnist/
wget -c "$URL"train-images-idx3-ubyte.gz
wget -c "$URL"train-labels-idx1-ubyte.gz
wget -c "$URL"t10k-images-idx3-ubyte.gz
wget -c "$URL"t10k-labels-idx1-ubyte.gz
