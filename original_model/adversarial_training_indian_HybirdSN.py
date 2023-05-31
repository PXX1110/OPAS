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
import h5py
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras.layers import Dense, Flatten, Conv2D, Conv3D, Dropout, Input, Reshape
# from keras.optimizer_v2.gradient_descent import SGD
from keras.optimizers import adam_v2
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras import backend as K
K.set_image_data_format('channels_first')
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

from sklearn.metrics import classification_report

from art.attacks.evasion import DeepFool
from art.attacks.evasion.pixel_threshold import PixelAttack
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.estimators.estimator import NeuralNetworkMixin
from art.estimators.classification import KerasClassifier


## Configure a logger to capture ART outputs; these are printed in console and the level of detail is set to INFO
# 配置记录器来捕获ART输出;这些信息会在控制台中打印出来，详细信息的级别设置为INFO
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
logger = logging.getLogger()
fh = logging.FileHandler('result/org/tests_tf_One_1_09.log')
formats = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formats)
logger.addHandler(fh)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(formats)

# Global Variables
# The number of principal components to be retained in the PCA algorithm, 
# the number of retained features  n
numPCAcomponents = 30
# Patches windows size
windowSize = 25
# The proportion of Test sets
testRatio = 0.90
# the names of features 
target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
           ,'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed', 
            'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
           'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
           'Stone-Steel-Towers']
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

# Define the input shape 
input_shape= X_train[0].shape
print(input_shape)
S = windowSize
L = numPCAcomponents
output_units =  16

## input layer
input_layer = Input((1, L, S, S))
conv_layer1 = Conv3D(filters=8, kernel_size=(7, 3, 3), activation='relu')(input_layer)
conv_layer2 = Conv3D(filters=16, kernel_size=(5, 3, 3), activation='relu')(conv_layer1)
conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)
print(conv_layer3.shape)
conv3d_shape = conv_layer3.shape
conv_layer3 = Reshape((conv3d_shape[1]*conv3d_shape[2], conv3d_shape[4], conv3d_shape[3]))(conv_layer3)
conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv_layer3)
flatten_layer = Flatten()(conv_layer4)
## fully connected layers
dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
dense_layer1 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.4)(dense_layer2)
output_layer = Dense(units=output_units, activation='softmax')(dense_layer2)
model = Model(inputs=input_layer, outputs=output_layer)

## Define optimization and train method
# keras_callbacks = [ModelCheckpoint(filepath="checkpoint/org/checkpoint_org_5309_One_HSN.hdf5", verbose=1, save_best_only=True),
#                   ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=25, min_lr=0.000001, verbose=1),
#                   CSVLogger('result/org/logs_One_1_09.csv', separator=';', append=True)]
# adam = adam_v2.Adam(lr=0.001, decay=1e-06)                             
# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

## load the model architecture and weights
model = load_model('checkpoint/org/checkpoint_org_5309_One_HSN.hdf5')

## Create classifier wrapper
classifier = KerasClassifier(model=model, clip_values=(min_, max_))

## Start to train model 
# history = model.fit(X_train, y_train, 
#                     batch_size=256, 
#                     epochs=100, 
#                     verbose=1, 
#                     validation_data=(X_test, y_test),
#                     callbacks=keras_callbacks,
#                     shuffle=True)
# model.save('./model/HSI_model_epochs100041701.h5')

## using plot_model module to save the model figure
# plot_model(model, to_file='./model/model.png', show_shapes=True)
# logger.info(history.history.keys())

## show the model figure
# model_img = plt.imread('./model/model.png')
# plt.imshow(model_img)
# plt.show()

# ## summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.grid(True)
# plt.legend(['train', 'test'], loc='upper left') 
# plt.savefig("./result/org/model_accuracy_5309_One_HSN.png")
# # plt.show()
# plt.close()  # 关闭窗口

# # ## summarize history for loss 
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.grid(True)
# plt.legend(['train', 'test'], loc='upper left') 
# plt.savefig("./result/org/model_loss_5309_One_HSN.png")
# # plt.show()
# plt.close()  # 关闭窗口

Y_pred = model.predict(X_test, batch_size=256)
y_pred = np.argmax(Y_pred, axis=1)
classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names, digits=4)
logger.info('Finished test')
logger.info("classification:{}".format(classification))

## Craft adversarial samples with DeepFool
# logger.info("Create DeepFool attack")
# adv_crafter = DeepFool(classifier)

# logger.info("Create PGD attack")
# adv_crafter = ProjectedGradientDescent(classifier, eps=8 / 255, eps_step=2 / 255, max_iter=10, num_random_init=1)

# logger.info("Create FGSM attack")
# NeuralNetworkMixin.channels_first = True
# adv_crafter = FastGradientMethod(classifier, eps=8/255, eps_step=2/255)

logger.info("Create PixelAttack attack")
NeuralNetworkMixin.channels_first = True
adv_crafter = PixelAttack(classifier, th=1)

traintic = time.time()
logger.info("Craft attack on training examples")
X_train = np.squeeze(X_train)                      # 删去维度是1的维度
x_train_adv = adv_crafter.generate(X_train)
traintoc = time.time()

logger.info("Craft attack test examples")
testtic = time.time()
X_test = np.squeeze(X_test)                        # 删去维度是1的维度
x_test_adv = adv_crafter.generate(X_test)
testtoc = time.time()

X_train = np.expand_dims(X_train, 1)
x_train_adv = np.expand_dims(x_train_adv, 1)
x_test_adv = np.expand_dims(x_test_adv, 1)

## Evaluate the classifier on the adversarial samples
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info("Classifier before adversarial training")
logger.info("Accuracy on adversarial samples: %.2f%%", (acc * 100))

## Data augmentation: expand the training set with the adversarial samples
x_train = np.append(X_train, x_train_adv, axis=0)
y_train = np.append(y_train, y_train, axis=0)

tf.reset_default_graph()
## Retrain the CNN on the extended dataset
keras_callbacks = [ModelCheckpoint(filepath="checkpoint/adv/checkpoint_adv_5309_One_HSN.hdf5", verbose=1, save_best_only=True),
                  ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=25, min_lr=0.000001, verbose=1),
                  CSVLogger('result/adv/logs_One_1_09.csv', separator=';', append=True)]
adam = adam_v2.Adam(lr=0.001, decay=1e-06)                             
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

## Create classifier wrapper
classifier = KerasClassifier(model=model, clip_values=(min_, max_))

## Start to train model 
advhistory = model.fit(x_train, y_train, 
                    batch_size=256, 
                    epochs=100, 
                    verbose=1, 
                    validation_data=(X_test, y_test),
                    callbacks=keras_callbacks,
                    shuffle=True)

# summarize history for accuracy
plt.imshow()
plt.plot(advhistory.history['accuracy'])
plt.plot(advhistory.history['val_accuracy'])
plt.title('adversarial trained model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid(True)
plt.legend(['train', 'test'], loc='upper left') 
plt.savefig("./result/adv/model_advaccuracy_5309_One_HSN.png")
# plt.show()
plt.close()  # 关闭窗口

# summarize history for loss 
plt.imshow()
plt.plot(advhistory.history['loss'])
plt.plot(advhistory.history['val_loss'])
plt.title('adversarial trained model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.grid(True)
plt.legend(['train', 'test'], loc='upper left') 
plt.savefig("./result/adv/model_advloss_5309_One_HSN.png")
# plt.show()
plt.close()  # 关闭窗口

## Evaluate the adversarially trained classifier on the test set
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info("Classifier with adversarial training")
logger.info("Accuracy on adversarial samples: %.2f%%", (acc * 100))


orgmodel = load_model('checkpoint/org/checkpoint_org_5309_One_HSN.hdf5')
orgloss,orgacc = orgmodel.evaluate(X_test,y_test,batch_size=256,verbose=0)
advmodel = load_model('checkpoint/adv/checkpoint_adv_5309_One_HSN.hdf5')
advloss,advacc = advmodel.evaluate(X_test,y_test,batch_size=256,verbose=0)
traintick_tock = traintic - traintoc
testtick_tock = testtic - testtoc
# logger.info('val_acc before adv = {:.4f}'.format(np.max(history.history['val_accuracy'])))
# logger.info('val_acc after adv = {:.4f}'.format(np.max(advhistory.history['val_accuracy'])))
logger.info('orgtest_loss:',orgloss,'orgtest_accuracy:',orgacc,'advtest_loss:',advloss,
      'advtest_accuracy:',advacc, 'advtraintime:', traintick_tock, 'adv testtime:', testtick_tock)