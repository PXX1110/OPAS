# -*- coding: utf-8 -*-
"""
Trains a convolutional neural network on the Indian_pines dataset, then generated adversarial images using the
DeepFool attack and retrains the network on the training set augmented with the adversarial images.
"""
## Configure
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import time
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras.layers import Dense, Flatten, Conv2D, Conv3D, Dropout, Input, Reshape
from keras import backend as K
K.set_image_data_format('channels_first')
from keras.utils import np_utils
from model.report import reports, plot_confusion_matrix
from dataset_position import loadData, applyPCA, createPatches, splitTrainTestSet
from art.attacks.evasion import ProjectedGradientDescent, CarliniL2Method, DeepFool, BasicIterativeMethod
# , AutoAttack
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.estimators.estimator import NeuralNetworkMixin
from art.estimators.classification import KerasClassifier
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# 'IP': 16; 'UP': 9; 'SA': 16; 
# Global Variables
# Dataset
dataset = 'IP'
# PCA components
numPCAcomponents = 30
# Patches windows size
windowSize = 25
# The proportion of Test sets
testRatio = 0.80   #09
testFile = 8
# Class
classes = 16
#Batch_size
batch_size = 256

## Configure a logger to capture ART outputs; these are printed in console and the level of detail is set to INFO
# 配置记录器来捕获ART输出;这些信息会在控制台中打印出来，详细信息的级别设置为INFO
logger = logging.getLogger()
fh = logging.FileHandler("result/tests_tf_" + str(testFile) + "_adversarial_attack_" + str(dataset) + ".log")
formats = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formats)
logger.addHandler(fh)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(formats)


# load Preprocessed data from file
X_train = np.load("./predata/XtrainWindowSize" 
                  + str(windowSize) + "PCA" + str(numPCAcomponents) + 
                  "testRatio" + str(testRatio)  + ".npy")
y_train = np.load("./predata/ytrainWindowSize" 
                  + str(windowSize) + "PCA" + str(numPCAcomponents) + 
                  "testRatio" + str(testRatio) + ".npy")
X_test = np.load("./predata/XtestWindowSize" 
                  + str(windowSize) + "PCA" + str(numPCAcomponents) + 
                 "testRatio" + str(testRatio)  + ".npy")
y_test = np.load("./predata/ytestWindowSize" 
                  + str(windowSize) + "PCA" + str(numPCAcomponents) + 
                 "testRatio" + str(testRatio) + ".npy")


# Reshape data into (numberofsumples,channels, 1, height, width) for model
X_train = np.reshape(X_train, (-1, 1, numPCAcomponents, windowSize, windowSize))
X_test = np.reshape(X_test, (-1, 1, numPCAcomponents, windowSize, windowSize))


# convert class labels to on-hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
min_ = np.min(X_train)
max_ = np.max(X_train)


## Define model
# Define the input shape 
input_shape= X_train[0].shape
S = windowSize
L = numPCAcomponents
output_units =  classes
# input layer
input_layer = Input((1, L, S, S))
conv_layer1 = Conv3D(filters=8, kernel_size=(7, 3, 3), activation='relu')(input_layer)
conv_layer2 = Conv3D(filters=16, kernel_size=(5, 3, 3), activation='relu')(conv_layer1)
conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)
conv3d_shape = conv_layer3.shape
conv_layer3 = Reshape((conv3d_shape[1]*conv3d_shape[2], conv3d_shape[4], conv3d_shape[3]))(conv_layer3)
conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv_layer3)
flatten_layer = Flatten()(conv_layer4)
# fully connected layers
dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
dense_layer1 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.4)(dense_layer2)
output_layer = Dense(units=output_units, activation='softmax')(dense_layer2)
model = Model(inputs=input_layer, outputs=output_layer)


## load the model architecture and weights
model = load_model("checkpoint/org/checkpoint_org_5308_One_HSN.hdf5")

## Create classifier wrapper
classifier = KerasClassifier(model=model, clip_values=(min_, max_))


logger.info("Create Adversarial attack")
NeuralNetworkMixin.channels_first = True
adv_crafters = [
    FastGradientMethod(estimator=classifier, batch_size=batch_size, eps=8/255, eps_step=2/255),
    BasicIterativeMethod(estimator=classifier, batch_size=batch_size, eps=8/255, eps_step=2/255, max_iter=10),
    CarliniL2Method(classifier=classifier, batch_size=batch_size, binary_search_steps=1, initial_const=1, max_iter=100),
    ProjectedGradientDescent(estimator=classifier, batch_size=batch_size, eps=8/255, eps_step=2/255, max_iter=10, num_random_init=1),
    # AutoAttack(estimator=classifier, batch_size=batch_size, eps=8/255),
    DeepFool(classifier=classifier, batch_size=batch_size)]
i = 0
for adv_crafter in adv_crafters:
    i += 1
    traintic = time.time()
    logger.info("Attack {} on training examples".format(i))
    x_train_adv = adv_crafter.generate(X_train, testFile=testFile, dataset=dataset)
    traintoc = time.time()
    traintick_tock = traintic - traintoc
    logger.info('advtraintime{:.4f} (sec)'.format(traintick_tock))


    testtic = time.time()
    logger.info("Attack {} on testing examples".format(i))             
    x_test_adv = adv_crafter.generate(X_test, testFile=testFile, dataset=dataset)
    testtoc = time.time()
    testtick_tock = testtic - testtoc
    logger.info('advtesttime{:.4f} (sec)'.format(testtick_tock))


    ## Evaluate the classifier on the adversarial samples
    classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(x_test_adv,y_test,model)
    classification = str(classification)
    each_acc = str(each_acc)
    logger.info("Classifier before adversarial training {}".format(i))
    logger.info('Start adversarial examples test')
    logger.info('Test loss on adversarial examples {:.4f} (%)'.format(Test_loss))
    logger.info('Test accuracy on adversarial examples {:.4f} (%)'.format(Test_accuracy))
    logger.info('OA {:.4f} (%)'.format(oa))
    logger.info('Each_Acc: ')
    logger.info('{}'.format(each_acc))
    logger.info('AA {:.4f} (%)'.format(aa))
    logger.info('Kappa {:.4f} (%)'.format(kappa))
    logger.info("classification result: ")
    logger.info('{}'.format(classification))
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(confusion, normalize=False, title='Confusion matrix on adversarial examples')
    plt.savefig("./result/adv/" + str(testFile) + "/confusion_matrix_adv_" + str(testFile) + 
                "_Attack{}_".format(i) + str(dataset) + ".png")
    # plt.show()
    plt.close()

    ## Define advmodel
    # input layer
    input_layer = Input((1, L, S, S))
    conv_layer1 = Conv3D(filters=8, kernel_size=(7, 3, 3), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16, kernel_size=(5, 3, 3), activation='relu')(conv_layer1)
    conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)
    conv3d_shape = conv_layer3.shape
    conv_layer3 = Reshape((conv3d_shape[1]*conv3d_shape[2], conv3d_shape[4], conv3d_shape[3]))(conv_layer3)
    conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv_layer3)
    flatten_layer = Flatten()(conv_layer4)
    # fully connected layers
    dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(units=output_units, activation='softmax')(dense_layer2)
    advmodel = Model(inputs=input_layer, outputs=output_layer)


    ## load the advmodel architecture and weights
    advmodel = load_model("checkpoint/adv/checkpoint_adv_5308_One_HSN.hdf5")


    ## Evaluate the adversarially trained classifier on the test set
    Adv_classification, Adv_confusion, Adv_Test_loss, Adv_Test_accuracy, oa, each_acc, aa, kappa = reports(x_test_adv,y_test,advmodel)
    Adv_classification = str(Adv_classification)
    each_acc = str(each_acc)
    logger.info("Classifier after adversarial training {}".format(i))
    logger.info('Adv test loss {:.4f} (%)'.format(Adv_Test_loss))
    logger.info('Adv test accuracy {:.4f} (%)'.format(Adv_Test_accuracy))
    logger.info('OA {:.4f} (%)'.format(oa))
    logger.info('Each_Acc: ')
    logger.info('{}'.format(each_acc))
    logger.info('AA {:.4f} (%)'.format(aa))
    logger.info('Kappa {:.4f} (%)'.format(kappa))
    logger.info("Adv classification result: ")
    logger.info('{}'.format(Adv_classification))
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(Adv_confusion, normalize=False, title='Confusion matrix after adversarial training')
    plt.savefig("./result/adv/" + str(testFile) + "/confusion_matrix_after_adv_" + str(testFile) + 
                "_Attack{}_".format(i) + str(dataset) + ".png")
    # plt.show()
    plt.close()
logger.info("Finish!")    