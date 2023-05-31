import numpy as np
import matplotlib.pyplot as plt
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion.pixel_threshold import PixelAttack
from model.common_tools import HybridSN, MakeDataset, ModelTrainer, plot_line, class_report
import time
import torch
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

if __name__ == '__main__':
    org_dir = "result/org"
    adv_dir = "result/adv"
    # 用于测试样本的比例
    testRatio = 0.90
    # 每个像素周围提取 patch 的尺寸
    windowSize = 25
    # 使用 PCA 降维，得到主成分的数量
    numPCAcomponents = 30
    S = windowSize
    L = numPCAcomponents
    Batch_Size = 256
    # milestones = [40, 60]
    MAX_EPOCH = 10
    ADV_MAX_EPOCH = 100

    Xtrain = np.load("./predata/XtrainWindowSize" 
                    + str(windowSize) + "PCA" + str(numPCAcomponents) + 
                    "testRatio" + str(testRatio)  + ".npy")
    ytrain = np.load("./predata/ytrainWindowSize" 
                    + str(windowSize) + "PCA" + str(numPCAcomponents) + 
                    "testRatio" + str(testRatio) + ".npy")
    Xtest = np.load("./predata/XtestWindowSize" 
                    + str(windowSize) + "PCA" + str(numPCAcomponents) + 
                    "testRatio" + str(testRatio)  + ".npy")
    ytest = np.load("./predata/ytestWindowSize" 
                    + str(windowSize) + "PCA" + str(numPCAcomponents) + 
                    "testRatio" + str(testRatio) + ".npy")

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    Xtrain = Xtrain.reshape(-1, windowSize, windowSize, numPCAcomponents, 1)
    Xtest  = Xtest.reshape(-1, windowSize, windowSize, numPCAcomponents, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape) 
    print('before transpose: Xtest  shape: ', Xtest.shape) 

    # 为了适应 pytorch 结构，数据要做 transpose
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2).astype(np.float32)
    Xtest  = Xtest.transpose(0, 4, 3, 1, 2).astype(np.float32)
    print('after transpose: Xtrain shape: ', Xtrain.shape) 
    print('after transpose: Xtest  shape: ', Xtest.shape) 

    min_pixel_value = np.min(Xtrain)
    max_pixel_value = np.max(Xtrain)

    # 创建 trainloader 和 testloader
    trainset = MakeDataset(Xtrain, ytrain)
    testset  = MakeDataset(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=Batch_Size, shuffle=True, num_workers=2)
    test_loader  = torch.utils.data.DataLoader(dataset=testset,  batch_size=Batch_Size, shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 网络放到GPU上
    model = HybridSN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # ============================  断点恢复 ============================
    # path_checkpoint = r"checkpoint/org/checkpoint_org042809_One_HSN.hdf5"
    # checkpoint = torch.load(path_checkpoint)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # # ============================  断点恢复 ============================

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, L, S, S),
        nb_classes=16,
    )

    # 开始训练
    logger.info('Start Training')
    start_epoch = -1
    loss_rec = []
    acc_rec = []
    for epoch in range(start_epoch + 1, MAX_EPOCH):
        loss_train, acc_train  = ModelTrainer.train(epoch, train_loader, model, criterion, optimizer, device) 
        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Train loss:{:.4f}".format(
                epoch + 1, MAX_EPOCH, acc_train, loss_train))
        # 保存模型
        best_acc = 0.92       # 随便设置一个比较大的数
        if acc_train > best_acc:
            best_acc = acc_train
            checkpoint = {"model_state_dict": model.state_dict()}
            orgpath_checkpoint = "checkpoint/org/checkpoint_org043009_One_HSN.pkl"
            torch.save(checkpoint, orgpath_checkpoint)
        # 绘图
        loss_rec.append(loss_train)
        acc_rec.append(acc_train)
        plt_x = np.arange(1, epoch+2)
        plot_line(plt_x, loss_rec, mode="loss_org_One_HSN", out_dir=org_dir)
        plot_line(plt_x, acc_rec, mode="acc_org_One_HSN", out_dir=org_dir)


    classification =  class_report(test_loader, ytest, model, device)
    logger.info(classification)
    ## Evaluate the ART classifier on adversarial test examples
    logger.info("Create PixelAttack attack")
    attack = PixelAttack(classifier, th=1)
    # attack = FastGradientMethod(estimator=classifier, eps=0.2, batch_size=Batch_Size)
    logger.info("Craft attack on training examples")

    traintic = time.time()
    Xtrain = np.squeeze(Xtrain) # 删去维度是1的维度
    Xtrain_adv = attack.generate(Xtrain)
    traintoc = time.time()

    # testtic = time.time()
    # Xtest = np.squeeze(Xtest) # 删去维度是1的维度
    # Xtest_adv = attack.generate(Xtest)
    # testtoc = time.time()

    Xtrain = np.expand_dims(Xtrain, 1)
    Xtrain_adv = np.expand_dims(Xtrain_adv, 1)
    # Xtest_adv = np.expand_dims(Xtest_adv, 1)

    # preds = np.argmax(classifier.predict(Xtest_adv,batch_size=320000), axis=1)
    # acc = np.sum(preds == np.argmax(ytest, axis=1)) / ytest.shape[0]

    x_train = np.append(Xtrain, Xtrain_adv, axis=0)
    y_train = np.append(ytrain, ytrain, axis=0)

    # 创建 trainloader 和 testloader
    advtrainset = MakeDataset(x_train, y_train)
    advtrain_loader = torch.utils.data.DataLoader(dataset=advtrainset, batch_size=Batch_Size, shuffle=True, num_workers=2)

    advoptimizer = optim.Adam(model.parameters(), lr=0.001)
    # # ============================  断点恢复 ============================
    # path_checkpoint = r"checkpoint/org/checkpoint_org042809_One_HSN.hdf5"
    # checkpoint = torch.load(path_checkpoint)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # # ============================  断点恢复 ============================

    # 开始对抗训练
    logger.info('Start AdvTraining')
    start_epoch = -1
    loss_rec = []
    acc_rec = []
    for epoch in range(start_epoch + 1, MAX_EPOCH):
        loss_train, acc_train = ModelTrainer.train(train_loader, model, criterion,optimizer, device) 
        loss_train_mean = np.mean(loss_train)
        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f}".format(
            epoch + 1, MAX_EPOCH, acc_train, loss_train_mean))
        # 绘图
        loss_rec.append(loss_train_mean)
        acc_rec.append(acc_train)
        plt_x = np.arange(1, epoch+2)
        plot_line(plt_x, loss_rec, mode="loss_adv_One_HSN", out_dir=org_dir)
        plot_line(plt_x, acc_rec, mode="acc_adv_One_HSN", out_dir=org_dir)
        # 保存模型
        best_acc = 0.92       # 随便设置一个比较大的数
        if acc_rec > best_acc:
            best_acc = acc_rec
            advcheckpoint = {"model_state_dict": model.state_dict()}
            orgpath_checkpoint = "checkpoint/org/checkpoint_org043009_One_HSN.pkl"
            torch.save(advcheckpoint, orgpath_checkpoint)

    advclassification =  class_report(test_loader, ytest, model, device)

    # preds = np.argmax(classifier.predict(Xtest_adv,batch_size=320000), axis=1)
    # advacc = np.sum(preds == np.argmax(ytest, axis=1)) / ytest.shape[0]

    # logger.info("Classifier before adversarial training")
    # logger.info("Accuracy on adversarial samples: %.2f%%", (acc * 100))

    # logger.info("Classifier after adversarial training")
    # logger.info("Accuracy on adversarial samples: %.2f%%", (advacc * 100))

    logger.info("classification:",classification,"advclassification:",advclassification,
            "advtraintime:",traintoc-traintic
            # ,testtoc-testtic
            )