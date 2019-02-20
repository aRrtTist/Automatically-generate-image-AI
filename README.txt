**********************工程环境**********************
OS：Windows 10 1809
CPU：Intel i5-6300HQ
GPU：NVIDIA GTX960M
Python：3.6.7
Tensorflow-gpu：1.8
cuda：9.0
cuDNN：7.0
**********************模块库**********************
os
scipy
glob
PIL
numpy
tensorflow
**********************代码说明**********************
本工程下各文件的功能如下：
images：训练所用的图片
generate：训练1000次后的模型输出样例
network.py：DCGAN 深层卷积生成对抗网络
train.py：训练 DCGAN
generate.py：用 DCGAN 的生成器模型 和 训练得到的生成器参数文件 来生成图片

