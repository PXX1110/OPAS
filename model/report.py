# coding: utf-8
# ## test.ipynb: Test the training result and Evaluate model
# Import the necessary libraries
import os
from re import L, T
from turtle import position
import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import itertools
import spectral
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from dataset_position import loadData


PATCH_SIZE = 25
# Get the model evaluation report, 
# include classification report, confusion matrix, Test_Loss, Test_accuracy
target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
           ,'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed', 
            'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
           'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
           'Stone-Steel-Towers']


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


## load the Indian pines dataset which is the .mat format
def loadIndianPinesData():
    data_path = os.path.join(os.getcwd(),'data')
    data = sio.loadmat(os.path.join(data_path, 'Indian_pines.mat'))['indian_pines']
    labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    
    return data, labels


## apply PCA preprocessing for data sets
def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca


##  pad zeros to dataset
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, 
                     X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + 
         y_offset, :] = X
    return newX


def Patch(data,height_index,width_index):
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    
    return patch


def reports(X_test, y_test, model):

    Y_pred = model.predict(X_test, batch_size=256)
    y_pred = np.argmax(Y_pred, axis=1)

    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names, digits=4)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    score = model.evaluate(X_test, y_test, batch_size=32)
    Test_Loss = score[0]*100
    Test_accuracy = score[1]*100
    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)

    return classification, confusion, Test_Loss, Test_accuracy, oa*100, each_acc*100, aa*100, kappa*100


# %matplotlib inline
def plot_confusion_matrix(cm, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.get_cmap("Blues")):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes = target_names
    Normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if normalize:
        cm = Normalized
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(Normalized, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        thresh = cm[i].max() / 2.
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


## calculate image
def calculate_image(model, dataset='IP', margin=PATCH_SIZE//2, adv=False, testFile=9):
    X, y = loadData(dataset)
    X, _ = applyPCA(X, numComponents=30)
    X = padWithZeros(X, margin)
    height = y.shape[0]
    width = y.shape[1]
    outputs = np.zeros((height,width))  
    for i in range(height):
        for j in range(width):
            target = int(y[i,j])
            if target == 0:
                continue
            else:
                image_patch=Patch(X,i,j)
                X_test_image = image_patch.reshape(-1, 1,image_patch.shape[2],image_patch.shape[0],image_patch.shape[1]).astype('float32')                                   
                prediction = (np.argmax(model.predict(X_test_image), axis=-1))                          
                outputs[i][j] = prediction+1
    if adv == True:
        positions = np.load("test_result/adv/" + str(testFile) + "/position_" + str(dataset) + ".npy") 
        classes = np.load("test_result/adv/" + str(testFile) + "/class_" + str(dataset) + ".npy") 
        for pos, cl in tqdm(zip(positions, classes)): 
            h = pos[0]
            l = pos[1]
            outputs[h-margin][l-margin] = cl+1
        spectral.save_rgb("test_result/adv/" + str(testFile) + "/attack_" + str(testFile) + "_One_1_HSN_position_" + str(dataset) + 
                          ".png", outputs.astype(int), colors=spectral.spy_colors)

    return outputs.astype(int)

 
    