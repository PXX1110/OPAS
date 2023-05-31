import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion.pixel_threshold import PixelAttack

import spectral
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class_num = 16

class HybridSN(nn.Module):
  def __init__(self):
    super(HybridSN, self).__init__()
    self.conv3d_1 = nn.Sequential(
        nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=1, padding=0),
        nn.BatchNorm3d(8),
        nn.ReLU(inplace = True),
    )
    self.conv3d_2 = nn.Sequential(
        nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=0),
        nn.BatchNorm3d(16),
        nn.ReLU(inplace = True),
    ) 
    self.conv3d_3 = nn.Sequential(
        nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0),
        nn.BatchNorm3d(32),
        nn.ReLU(inplace = True)
    )

    self.conv2d_4 = nn.Sequential(
        nn.Conv2d(576, 64, kernel_size=(3, 3), stride=1, padding=0),
        nn.BatchNorm2d(64),
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

# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    return X_train, X_test, y_train, y_test

# 地物类别
class_num = 16
X = sio.loadmat('data\Indian_pines_corrected.mat')['indian_pines_corrected']
y = sio.loadmat('data\Indian_pines_gt.mat')['indian_pines_gt']

# 用于测试样本的比例
test_ratio = 0.90
# 每个像素周围提取 patch 的尺寸
patch_size = 25
# 使用 PCA 降维，得到主成分的数量
pca_components = 30

S = patch_size
L = pca_components

Batch_Size = 256

print('Hyperspectral data shape: ', X.shape)
print('Label shape: ', y.shape)

print('\n... ... PCA tranformation ... ...')
X_pca = applyPCA(X, numComponents=pca_components)
print('Data shape after PCA: ', X_pca.shape)

print('\n... ... create data cubes ... ...')
X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
print('Data cube X shape: ', X_pca.shape)
print('Data cube y shape: ', y.shape)

print('\n... ... create train & test data ... ...')
Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)
print('Xtrain shape: ', Xtrain.shape)
print('Xtest  shape: ', Xtest.shape)

# 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
Xtest  = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
print('before transpose: Xtrain shape: ', Xtrain.shape) 
print('before transpose: Xtest  shape: ', Xtest.shape) 

# 为了适应 pytorch 结构，数据要做 transpose
Xtrain = Xtrain.transpose(0, 4, 3, 1, 2).astype(np.float32)
Xtest  = Xtest.transpose(0, 4, 3, 1, 2).astype(np.float32)
print('after transpose: Xtrain shape: ', Xtrain.shape) 
print('after transpose: Xtest  shape: ', Xtest.shape) 

min_pixel_value = np.min(Xtrain)
max_pixel_value = np.max(Xtrain)

""" Training dataset"""
class TrainDS(torch.utils.data.Dataset): 
    def __init__(self):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)        
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len

""" AdvTraining dataset"""
class AdvTrainDS(torch.utils.data.Dataset): 
    def __init__(self):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)        
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len

""" Testing dataset"""
class TestDS(torch.utils.data.Dataset): 
    def __init__(self):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len

# 创建 trainloader 和 testloader
trainset = TrainDS()
testset  = TestDS()
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=Batch_Size, shuffle=True, num_workers=0)
test_loader  = torch.utils.data.DataLoader(dataset=testset,  batch_size=Batch_Size, shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 网络放到GPU上
model = HybridSN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, L, S, S),
    nb_classes=16,
)

# 开始训练
total_loss = 0
for epoch in range(10):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 优化器梯度归零 
        optimizer.zero_grad()
        # 正向传播 +　反向传播 + 优化 
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' %(epoch + 1, total_loss/(epoch+1), loss.item()))

print('Finished Training')

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
print(classification)

# ## Step 7: Evaluate the ART classifier on adversarial test examples
# attack = PixelAttack(classifier, th=1)
# # attack = FastGradientMethod(estimator=classifier, eps=0.2, batch_size=Batch_Size)
# Xtrain = np.squeeze(Xtrain) # 删去维度是1的维度
# Xtrain_adv = attack.generate(Xtrain)
# Xtest = np.squeeze(Xtest) # 删去维度是1的维度
# Xtest_adv = attack.generate(Xtest)

# Xtrain = np.expand_dims(Xtrain, 1)
# Xtrain_adv = np.expand_dims(Xtrain_adv, 1)
# Xtest_adv = np.expand_dims(Xtest_adv, 1)

# x_train = np.append(Xtrain, Xtrain_adv, axis=0)
# y_train = np.append(ytrain, ytrain, axis=0)

# # 创建 trainloader 和 testloader
# advtrainset = AdvTrainDS()
# advtrain_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=Batch_Size, shuffle=True, num_workers=0)

# # 开始训练
# total_loss = 0
# for epoch in range(10):
#     for i, (inputs, labels) in enumerate(advtrain_loader):
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         # 优化器梯度归零
#         optimizer.zero_grad()
#         # 正向传播 +　反向传播 + 优化 
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' %(epoch + 1, total_loss/(epoch+1), loss.item()))

# print('Finished Training')

# count = 0
# # 模型测试
# for inputs, _ in test_loader:
#     inputs = inputs.to(device)
#     outputs = model(inputs)
#     outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
#     if count == 0:
#         y_pred_test =  outputs
#         count = 1
#     else:
#         y_pred_test = np.concatenate( (y_pred_test, outputs) )

# # 生成分类报告*/*
# classification = classification_report(ytest, y_pred_test, digits=4)
# print(classification)