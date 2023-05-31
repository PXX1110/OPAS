# -*- coding: utf-8 -*-
"""
# @file name  : common_tools.py
# @author     : PengXiaoxu https://github.com/PengXiaoxu
# @date       : 2022-04-28
# @brief      : 通用函数
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import time
import json
import matplotlib.pyplot as plt
import torchvision.models as models
from sklearn.metrics import classification_report

class HybridSN(nn.Module):
    def __init__(self):
        super(HybridSN, self).__init__()
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=1, padding=0),
            nn.ReLU(inplace = True),
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=0),
            nn.ReLU(inplace = True),
        ) 
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.ReLU(inplace = True)
        )

        self.conv2d_4 = nn.Sequential(
            nn.Conv2d(576, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(inplace = True),
        )
        self.fc1 = nn.Linear(18496,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,16)
        self.dropout = nn.Dropout(p = 0.4)

    def forward(self,x):
        out = self.conv3d_1(x)
        out = self.conv3d_2(out)
        out = self.conv3d_3(out)
        out = self.conv2d_4(out.reshape(out.shape[0],-1,19,19))
        out = out.reshape(out.shape[0],-1)
        out = F.relu(self.dropout(self.fc1(out)))
        out = F.relu(self.dropout(self.fc2(out)))
        out = self.fc3(out)
        return out

""" Makeing dataset"""
class MakeDataset(torch.utils.data.Dataset): 
    def __init__(self, Xtrain, ytrain):
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.len = self.Xtrain.shape[0]
        self.x_data = torch.FloatTensor(self.Xtrain)
        self.y_data = torch.LongTensor(self.ytrain)        
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len


class ModelTrainer(object):

    @staticmethod
    def train(epoch, data_loader, model, loss_f, optimizer, device):
        total_loss = 0
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()
            # 统计预测信息
            _, predicted = torch.max(outputs.data, 1)
            # 统计loss, acc
            Loss = loss.item()
            total_loss += loss.item()
            acc_avg = (predicted.cpu() == labels.cpu()).type(torch.float).sum().item() / labels.shape[0]
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' %(epoch + 1, total_loss/(epoch+1), loss.item()))
        return Loss, acc_avg

    @staticmethod
    def valid(data_loader, model, loss_f, device):

        model.eval()
        loss_sigma = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(data_loader):

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_f(outputs, labels)
                # 统计预测信息
                _, predicted = torch.max(outputs.data, 1)
                # 统计loss, acc
                loss_sigma.append(loss.item())
                acc_avg = (predicted.cpu() == labels.cpu()).type(torch.float).sum().item() / labels.shape[0]
            
        return loss_sigma, acc_avg


def show_confMat(confusion_mat, classes, set_name, out_dir, verbose=False):
    """
    混淆矩阵绘制
    :param confusion_mat:
    :param classes: 类别名
    :param set_name: trian/valid
    :param out_dir:
    :return:
    """
    cls_num = len(classes)
    # 归一化
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 显示

    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix' + set_name + '.png'))
    # plt.show()
    plt.close("all")
    # plt.close()

    if verbose:
        for i in range(cls_num):
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :])),
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i]))))

def plot_line(train_x, train_y, mode, out_dir):
    """
    绘制训练和验证集的loss曲线/acc曲线
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    plt.plot(train_x, train_y, label='Train')
    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close("all")
    # plt.close()

def plot_line_org(train_x, train_y, valid_x, valid_y, mode, out_dir):
    """
    绘制训练和验证集的loss曲线/acc曲线
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close("all")
    # plt.close()

def plot_line_adv(train_x, train_y, valid_x, valid_y, mode, out_dir):
    """
    绘制训练和验证集的loss曲线/acc曲线
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    plt.plot(train_x, train_y, label='Adv_Train')
    plt.plot(valid_x, valid_y, label='Adv_Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close("all")
    # plt.close()

def class_report(test_loader, ytest, model, device):
    count = 0
    # 模型测试
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test =  outputs
            count = 1
        else:
            y_pred_test = np.concatenate( (y_pred_test, outputs) )

    # 生成分类报告*/*
    classification = classification_report(ytest, y_pred_test, digits=4)
    
    return classification