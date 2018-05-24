#!/bin/bash

wget -c https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
mv iris.data iris.csv
sed -i -e 's/Iris-setosa/0/g' -e 's/Iris-versicolor/1/g' -e 's/Iris-virginica/2/g' iris.csv
