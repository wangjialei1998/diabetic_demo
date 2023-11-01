# 这是一个在糖尿病视网膜数据集eyepacs上进行训练的demo
这个项目用于示例如何在eyepacs上跑一个深度学习模型并进行预测，建议最好在linux服务器上进行尝试比较方便，如果是在自己的电脑上，则需要上网找找如何安装[Anaconda](https://www.anaconda.com/)、[Cuda](https://developer.nvidia.com/cuda-toolkit)、[Cudnn](https://developer.nvidia.com/cudnn)、Pytorch（在anaconda上安装）
## 基本信息
1. 使用模型resnet50

2. 数据集eyepacs



3. 框架：pytorch

4. 在gpu上进行训练和预测



## 具体步骤
1. 安装必要的驱动和软件（具体安装方法google 百度）
   >cuda (非必要，如果没有就在cpu上面进行运算，很慢)
   >
   >cudnn(非必要，如果没有就在cpu上面进行运算，很慢)
   >
   >anaconda（必要，包管理器，能创建独立的环境）
2. 在当前目录下执行以下命令安装必要的包和模块
   ```
   pip install -r requirements.txt
   #若没有安装pip则执行先在conda中安装pip
   ```e
   
3. 首先需要下载eyepacs数据集，下载网址 <cite>https://www.kaggle.com/c/diabetic-retinopathy-detection/

4. 下载后我们假设解压到了这个目录<strong>`/path/to/eyepacs`</strong>(后面都将使用这个作为eyepacs数据集的示例目录，需要按实际要求进行修改)
   ```
   我们假设数据集的结构和名字是这样的，如果不一样请将你的eyepacs数据集改成一样的名字！！！只需要确定四个地方，训练图像文件夹名称(改为train)、测试图像文件夹名称(改为test)、训练标签文件名称(改为train.csv)、测试标签文件名称(改为test.csv)

   eyepacs_dataset
   |
   |-train
   | |-1_left.jpeg
   | |-2_right.jpeg
   | |...
   |
   |-test
   | |-1_left.jpeg
   | |-2_right.jpeg
   | |...
   |
   |-train.csv
   |
   |-test.csv
   ```

5. 在这个demo项目的文件夹下面运行以下命令进行训练(make是一个linux上的构建工具，具体的命令写在项目目录Makefile中，如果是在window上可以直接运行下面的替代命令)
    ```
    make train
    #若是在window中运行，则直接在项目目录运行下面的命令
    ``` 
6. 在这个demo项目的文件夹下面运行以下命令进行预测
    ```
    make predict
    #若是在window中运行，则直接在项目目录运行下面的命令
    ```

## 一些小建议
1. 刚开始看深度学习的代码可能很难看懂，可以在b站先学学这个课程<cite>https://space.bilibili.com/1567748478/channel/seriesdetail?sid=358497</cite>，跟着一步步走走，就入门了

2. 对于demo中不了解的地方，直接复制代码问chatgpt，肯定就知道用法了