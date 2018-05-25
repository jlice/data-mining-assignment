# data-mining-assignment

这是《程序设计方法学/数据挖掘》的课程作业。作业的要求是不使用第三方库和标准库，编程语言不使用MATLAB，实现一个数据挖掘算法。

本项目是使用Python实现一个简单的神经网络，并在iris和MNIST数据集上进行了测试。

### 使用

下载本项目：

``` Bash
$ git clone https://github.com/jlice/data-mining-assignment.git
$ cd data-mining-assignment
```

准备数据集：

``` Bash
$ pip3 install numpy
$ cd MNIST-data
$ bash ./download.sh
$ python3 ./convert_to_csv.py
$ cd ..
```

训练：

``` Bash
$ python3 ./main.py
```

### 项目结构

```
.
├── iris                    鸢尾花数据集
│   ├── download.sh         下载数据集
│   └── iris.csv            已下载的数据集
├── MNIST-data              手写数字数据集
│   ├── convert_to_csv.py   转为csv
│   ├── download.sh         下载数据集
├── data.py                 数据集加载
├── layer.py                神经网络中的层
├── main.py                 训练主程序
├── matrix.py               矩阵相关计算封装
├── rand.py                 随机数生成
├── README.md               说明文档
└── util.py                 神经网络实用函数
```
