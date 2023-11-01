#这个直接用train.py中的validate_process就可以
import os
# 用于传命令行参数
import argparse
import dataset
import numpy as np
# pytorch相关的模块，比较多，建议看看pytorch的官方文档
import torch
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
# 一个模型库，可以直接获得常用的模型以及权重
import timm
# 用于cuda训练减少使用资源的，感兴趣可以查一查cuda单精度，自动混合精度等的相关东西，可以不用这个，这个demo展示一下，可以删去相关代码用全精度的
from torch.cuda.amp import GradScaler, autocast
from train import evaluate_metrix
parser = argparse.ArgumentParser(description='pytorh diabetic demo')
parser.add_argument('--num-classes', default=5)
# 用于指定dataloader开多少个进程进行加载数据，一般是4、8、16左右
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers')
# 图片缩放大小，由于深度学习网络要求输入的图像大小是一样的，所以需要指定一个统一大小，这里默认448
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 448)')
# 一次训练的批大小，这个要看显存大小，越大越快，但是可能会导致显存溢出报错
parser.add_argument('--batch-size', default=320, type=int,
                    metavar='N', help='mini-batch size')
# 这里传的是eyepacs数据集目录路径，需要修改
parser.add_argument("--data-dir", default="/path/to/eyepacs/dataset",
                    type=str, help="需要指定eyepacs数据集的路径哦！")
# 这个是用于预测的模型的路径，需要指定
parser.add_argument("--checkpoint", default="/path/to/eyepacs/dataset",
                    type=str, help="用于预测的模型的路径")
def main():
    args = parser.parse_args()
    # 创建模型,这里创建的是resnet50模型,并且加载了别人预先训练好的模型权重,如果需要改模型可以直接改第一个参数resnet50
    model = timm.create_model(
        "resnet50", pretrained=False, num_classes=args.num_classes)
    # 这个是模型的权重，这里严格加载到模型中去
    state_dict=torch.load(args.checkpoint,map_location='cpu')
    model.load_state_dict(state_dict,strict=True)
    print("create model {}".format("resnet50"))
    use_cuda=torch.cuda.is_available()
    if use_cuda:
        # 这个将模型转换到gpu的内存中
        model = model.cuda()
        print("模型将在gpu上进行训练")
    else:
        print("gpu不可用，模型将在cpu上进行训练")
    # 构造eyepacs的训练和测试图像位置以及标签文件的位置,这里需要修改，因为训练集文件夹名称可能不知道

    # 由于eyepacs没有指定验证数据集，这里直接使用测试数据集来充当验证数据集，跑起来之后你可以自己将训练数据集分成
    # 验证和训练两部分，减少验证的时间，因为验证是每个epoch都需要的
    test_imgs_dir = os.path.join(args.data_dir, "test")
    test_csv = os.path.join(args.data_dir, "test.csv")
    test_dataset = dataset.EyepacsImageDataset(imgs_dir=test_imgs_dir, labels_csv=test_csv, transform=transforms.Compose([
        # 这个就是将图片大小统一成一个image_size*image_size大小的函数
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ]))
    print("len(test_dataset)): ", len(test_dataset))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    eval(model,5,test_loader)
    
# 评估函数，输出产生的预测标签和内容，并且保存在根目录上
def eval(model,n_class,test_loader):
    print("starting predicting")
    Sig = torch.nn.Sigmoid()
    # 记录指标，这个是记录正确的个数
    running_corrects = 0.0
    # 下面的是常用指标，用于求混淆矩阵的，可以查查具体是怎么计算的，后面也有计算过程可以看看
    FM = torch.zeros((n_class, n_class))
    tp = torch.zeros(n_class)
    tn = torch.zeros(n_class)
    fp = torch.zeros(n_class)
    fn = torch.zeros(n_class)
    use_cuda=torch.cuda.is_available()
    model.eval()
    for i, (input, target) in enumerate(test_loader):
        if use_cuda:
            target = target.cuda()
            input=input.cuda()
        # 和训练差不多，但是不需要记录梯度了
        with torch.no_grad():
            if use_cuda:
                with autocast():
                    output= model(input)
            else:
                output= model(input)
            output=Sig(output).cpu()
        _, preds = torch.max(output.data, 1)
        running_corrects += torch.sum(preds.cpu() == target.cpu()).item()
        target = target.cpu()
        preds = preds.cpu()
        # 下面的for循环就是用来求相关指标的，也不需要看
        for batch_i in range(len(target)):
            # 求混淆矩阵的相关的东西
            predict_label = preds[batch_i]
            true_label = target[batch_i]
            #FM[predict_label][true_label] = FM[predict_label][true_label] + 1
            FM[true_label][predict_label] = FM[true_label][predict_label] + 1
            for label in range(n_class):
                p_or_n_from_pred = (label == preds[batch_i])
                p_or_n_from_label = (label == target[batch_i])
                if p_or_n_from_pred == 1 and p_or_n_from_label == 1:
                    tp[label] += 1
                if p_or_n_from_pred == 0 and p_or_n_from_label == 0:
                    tn[label] += 1
                if p_or_n_from_pred == 1 and p_or_n_from_label == 0:
                    fp[label] += 1
                if p_or_n_from_pred == 0 and p_or_n_from_label == 1:
                    fn[label] += 1
        evaluate_metrix("eval",n_class,0,0,tp,tn,fn,fp,len(test_loader.dataset),FM,0,running_corrects)
if __name__ == '__main__':
    main()
