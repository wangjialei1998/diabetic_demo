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

parser = argparse.ArgumentParser(description='pytorh diabetic demo')
parser.add_argument('--lr', default=1e-4, type=float)
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

# 这个用于记录目前最高的准确率
def main():
    # 获取命令行的参数
    args = parser.parse_args()

    # 默认将模型权重保存在models文件夹下面
    if not os.path.exists("./models"):
        os.mkdir("./models")
    else:
        print("models文件夹已经存在，请确保里面的内容是不需要的，训练过程将会覆盖其中的文件，若文件需要请ctrl+z终止程序")
    # 创建模型,这里创建的是resnet50模型,并且加载了别人预先训练好的模型权重,如果需要改模型可以直接改第一个参数resnet50
    model = timm.create_model(
        "resnet50", pretrained=True, num_classes=args.num_classes)
    print("create model {}".format("resnet50"))
    # 判断gpu是否可用，如果不可用就在cpu上训练
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
    val_imgs_dir = os.path.join(args.data_dir, "test")
    val_csv = os.path.join(args.data_dir, "test.csv")
    train_imgs_dir = os.path.join(args.data_dir, "train")
    train_csv = os.path.join(args.data_dir, "train.csv")

    # 这里加载训练数据集和验证数据集
    # 这个是验证数据集
    # 这里的transform使用的是pytorch提供的torchvision中的包的，查查文档，还有其他很多的变换！！注意的是最后一定是变成tensor，也就是pytorch能处理的数据结构，在demo中是totensor()函数的作用
    val_dataset = dataset.EyepacsImageDataset(imgs_dir=val_imgs_dir, labels_csv=val_csv, transform=transforms.Compose([
        # 这个就是将图片大小统一成一个image_size*image_size大小的函数
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ]))
    train_dataset = dataset.EyepacsImageDataset(imgs_dir=train_imgs_dir, labels_csv=train_csv, transform=transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        # 我在训练集中加多了一个随机水平反转的变换，你也可以加多别的来提高数据集的多样性
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ]))
    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))

    # 这个是dataloader，就是数据加载器，后面循环训练的时候就是通过train_loader来不断加载图片数据到模型中，具体的参数可以参考pytorch的官方文档
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

# 这个就是训练过程了
    train_process(model, train_loader, val_loader, args.lr)


def train_process(model, train_loader, val_loader, lr, n_class=5):
    best_acc=0.0
    # 设置epoch的次数的，我就先设个40,可以改为其他的，看那个效果好
    Epochs = 40
    # 权重递减参数，不懂的话看看深度学籍基础
    weight_decay = 1e-4
    # 损失函数，这里就直接使用crossentropy，就是最常见的
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器，这里使用adam
    optimizer = torch.optim.Adam(params=model.parameters(
    ), lr=lr, weight_decay=weight_decay)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    # 这个是训练计划器，可以按照epoch来调整训练的超参数，有多种，具体看文档
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)

    scaler = GradScaler()
    # 这个用来指示是否使用cuda
    use_cuda = torch.cuda.is_available()
    for epoch in range(Epochs):
        # 记录指标，这个是记录正确的个数
        running_corrects = 0.0
        running_loss = 0.0
        # 下面的是常用指标，用于求混淆矩阵的，可以查查具体是怎么计算的，后面也有计算过程可以看看
        FM = torch.zeros((n_class, n_class))
        tp = torch.zeros(n_class)
        tn = torch.zeros(n_class)
        fp = torch.zeros(n_class)
        fn = torch.zeros(n_class)

        # 将模型调整至训练状态
        model.train()
        for i, (inputData, target) in enumerate(train_loader):
            if use_cuda:
            # 输入的一批图像数据
                inputData = inputData.cuda()
                # 相对应的病变标签数据
                target = target.cuda()
            if use_cuda:
                with autocast():  # mixed precision
                    # 图片经过模型产生结果output，就是5个病变等级的概率
                    output = model(inputData).float()
                # 计算损失
                loss = criterion(output, target)
                # 进行反向传播
                scaler.scale(loss).backward()
                # 清除模型网络的梯度，因为每个节点都会保留上一个循环计算的梯度，不清除就会累积起来
                model.zero_grad()
                # 优化器更新
                scaler.step(optimizer)
                scaler.update()
            else:
                # 下面是一样的，只不过使用的是cpu来计算，略有不同
                output = model(inputData).float()
                loss = criterion(output, target)
                loss.backward()
                model.zero_grad()
                optimizer.step()
            # 计划器更新
            scheduler.step()
            # 获得最大值的下标，用于计算正确率
            _, preds = torch.max(output.data, 1)
            running_loss+=loss.sum().detach().cpu().numpy()
            running_corrects += torch.sum(preds.cpu() == target.cpu()).item()
            # 这里更新指标的数据，有点多，刚开始可以直接略过
            target = target.cpu()
            preds = preds.cpu()
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
            # 用于每100个batch输出一下现在的状态
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0],
                              loss.item()))
        # 这个用于在命令行打印训练信息和保存信息到文件中去
        evaluate_metrix("train",n_class,epoch,Epochs,tp,tn,fn,fp,len(train_loader.dataset),FM,running_loss,running_corrects)
        # 每5个epoch保存一次模型权重，这个可以改
        if epoch % 5 == 0:
            try:
                # 将模型保存在models的文件夹下面
                torch.save(model.state_dict(), os.path.join(
                    'models/', 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
            except:
                pass
        # 将模型调整至预测状态，会快很多，因为不需要记录梯度相关的信息
        model.eval()
        # 获得现在的acc，如果比历史记录的高就保存下来
        now_acc=validate_process(val_loader=val_loader,model=model,n_class=n_class,criterion=criterion)
        if now_acc > best_acc:
            best_acc=now_acc
            try:
                torch.save(model.state_dict(), os.path.join(
                    'models/', 'model-best.ckpt'))
            except:
                pass
        print('current_acc = {:.2f}, highest_acc = {:.2f}\n'.format(
            now_acc,best_acc))


def validate_process(val_loader, model,n_class,criterion):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    # 记录指标，这个是记录正确的个数
    running_corrects = 0.0
    running_loss = 0.0
    # 下面的是常用指标，用于求混淆矩阵的，可以查查具体是怎么计算的，后面也有计算过程可以看看
    FM = torch.zeros((n_class, n_class))
    tp = torch.zeros(n_class)
    tn = torch.zeros(n_class)
    fp = torch.zeros(n_class)
    fn = torch.zeros(n_class)
    use_cuda=torch.cuda.is_available()
    for i, (input, target) in enumerate(val_loader):
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
            loss = criterion(output, target)
            output=Sig(output).cpu()
        _, preds = torch.max(output.data, 1)
        running_loss+=loss.sum().detach().cpu().numpy()
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
    now_acc=evaluate_metrix("eval",n_class,0,0,tp,tn,fn,fp,len(val_loader.dataset),FM,running_loss,running_corrects)
    return now_acc


# 这个函数就是求相关指标的，sklearn那些包也有实现的，不需要看hhhh
# 这个函数用于计算相关的指标，并且保存在文件中，
def evaluate_metrix(train_or_eval,n_class,epoch,Epochs,tp,tn,fn,fp,dataset_len,FM,running_loss,running_corrects):
    precision = [0] * n_class
    recall = [0] * n_class
    specificity = [0] * n_class
    f1 = [0] * n_class
    # 求相关指标,并且保存在models文件夹下面
    for label in range(n_class):
        precision[label] = tp[label] / (tp[label] + fp[label] + 1e-8)
        recall[label] = tp[label] / (tp[label] + fn[label] + 1e-8)
        specificity[label] = tn[label] / (tn[label] + fp[label] + 1e-8)
        f1[label] = 2 * precision[label] * recall[label] / \
            (precision[label] + recall[label] + 1e-8)
        print('Epoch [{}/{}] {}_Data:\tClass {}:  Precision: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, F1:{:.4f}'.format(
            epoch, Epochs - 1, "train", label, precision[label], recall[label], specificity[label], f1[label]))
        with open(f'models/model_{train_or_eval}_result.txt', 'a+') as fileHandle:
            fileHandle.write('Epoch [{}/{}] {}_Data:\tClass {}:  Precision: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, F1:{:.4f}\n'.format(
                epoch, Epochs, "train", label, precision[label], recall[label], specificity[label], f1[label]))
    # 保存混淆矩阵，这个可以查查是什么
    print('{}_Data:Fusion Matrix:'.format(train_or_eval))
    print(np.array(FM))
    with open(f'models/model_{train_or_eval}_result.txt', 'a+') as fileHandle:
        fileHandle.write('\nFusion Matrix:\n')
        for f_i in np.array(FM.cpu()):
            fileHandle.write(str(f_i)+'\r\n')
        fileHandle.write('\n')
    # 计算kappa值
    pes = (tp+fn)*(tp+fp)
    pe = pes.sum().item() / (dataset_len*dataset_len)
    pa = tp.sum().item()/dataset_len
    kappa = (pa - pe) / (1 - pe)
    epoch_loss = running_loss / dataset_len
    epoch_acc = running_corrects / dataset_len
    epoch_kappa = kappa
    print('Epoch [{}/{}] {} loss: {:.4f} acc: {:.4f} kappa: {:.4f}\n'.format(
                epoch, Epochs,train_or_eval,
                epoch_loss, epoch_acc, epoch_kappa))
    # save the train result
    with open(f'models/model_{train_or_eval}_result.txt', 'a') as fileHandle:
        fileHandle.write('Epoch [{}/{}] {} loss: {:.4f} acc: {:.4f} kappa: {:.4f}\n'.format(
                                epoch,Epochs,train_or_eval,
                                epoch_loss, epoch_acc, epoch_kappa))
    return epoch_acc
if __name__ == '__main__':
    main()
