import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# 这个是数据集的定义，简单说就是告诉框架怎么取图片，相应的分级标签在哪里，具体可以看pytorch文档，一般使用只需要实现__len__()和__getitem__()这两个接口就好
class EyepacsImageDataset(Dataset):
    def __init__(self, imgs_dir: str, labels_csv: str, transform=None) -> None:
        super().__init__()
        # 获取图像所在的文件夹
        self.imgs_dir = imgs_dir
        # 标签文件
        self.labels_csv = labels_csv
        self.transform = transform
        self.data_list = pd.read_csv(self.labels_csv)
    # 用于dataloader模块获取数据集的总长度
    def __len__(self):
        return len(self.data_list)
    # 用于dataloader模块根据序号来获取每一个图片以及对应的标签
    def __getitem__(self, index):
        # 获取图片文件名和对应的病变等级
        img_name, level = self.data_list.iloc[index]['id'], self.data_list.iloc[index]['level']
        # 根据csv表格获取文件的名称
        img_name = '{}.{}'.format(img_name,"jpeg")
        # 构建文件路径
        img_path = os.path.join(self.imgs_dir, img_name)
        # 读取文件
        image = Image.open(img_path)
        # 判断文件是否存在,不存在就报错
        if image is None:
            print('image error ', img_path, "is not exist!")
            raise ValueError("image error ", img_path, "is not exist!")
        # 若有对图像的变换,就使用这个函数,transform是传入的函数,能直接调用,例如随机旋转图片那些
        if self.transform:
            image = self.transform(image)
        # 返回最后的图片和病变等级
        return image, level
