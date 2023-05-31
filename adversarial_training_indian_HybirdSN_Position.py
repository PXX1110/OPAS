# -*- coding: utf-8 -*-
"""
Trains a convolutional neural network on the Indian_pines dataset, then generated adversarial images using the
DeepFool attack and retrains the network on the training set augmented with the adversarial images.
"""
## Configure
from __future__ import absolute_import, division, print_function, unicode_literals
from doctest import testfile
import logging
import numpy as np
import time
import spectral
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras.layers import Dense, Flatten, Conv2D, Conv3D, Dropout, Input, Reshape
# from keras.optimizers import Adam
from keras.optimizers import adam_v2
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras import backend as K
K.set_image_data_format('channels_first')
from keras.utils import np_utils
from model.report import reports, plot_confusion_matrix, calculate_image
from dataset_position import loadData, applyPCA, createPatches, splitTrainTestSet
from art.attacks.evasion.pixel_threshold_position import PixelAttackPosition
from art.estimators.estimator import NeuralNetworkMixin
from art.estimators.classification import KerasClassifier
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# 'IP': 16; 'UP': 9; 'SA': 16; 
# Global Variables
numPCAcomponents = 30
# Patches windows size
windowSize = 25
# The proportion of Test sets
testRatio = 0.90   #09
testFile = 9
# Dataset
dataset = 'IP'
# Classes
classes = 16


## Configure a logger to capture ART outputs; these are printed in console and the level of detail is set to INFO
# 配置记录器来捕获ART输出;这些信息会在控制台中打印出来，详细信息的级别设置为INFO
logger = logging.getLogger()
fh = logging.FileHandler('result/tests_tf_'+ str(testFile) +'_One_1_HSN_position.log')
formats = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formats)
logger.addHandler(fh)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(formats)
 

# load Preprocessed data from file
X, y = loadData(dataset)
X, pca = applyPCA(X, numComponents = numPCAcomponents)
# Preprocess Data
XALL, yPatches = createPatches(X, y, windowSize=windowSize)
XTRAIN, XTEST, y_train, y_test = splitTrainTestSet(XALL, yPatches, y.max()-y.min(), testRatio)


# Reshape data into (numberofsumples,channels, 1, height, width) for model
XTRAIN["image"][0] = np.reshape(XTRAIN["image"][0], (-1, 1, numPCAcomponents, windowSize, windowSize))
XTEST["image"][0] = np.reshape(XTEST["image"][0], (-1, 1, numPCAcomponents, windowSize, windowSize))
X_train = XTRAIN["image"][0]
X_test = XTEST["image"][0] 
# convert class labels to on-hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
min_ = np.min(X_train)
max_ = np.max(X_train)


# Define the input shape 
input_shape= X_train[0].shape
S = windowSize
L = numPCAcomponents
output_units =  classes
## input layer
input_layer = Input((1, L, S, S))
conv_layer1 = Conv3D(filters=8, kernel_size=(7, 3, 3), activation='relu')(input_layer)
conv_layer2 = Conv3D(filters=16, kernel_size=(5, 3, 3), activation='relu')(conv_layer1)
conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)
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


## load the model architecture and weights
model = load_model('checkpoint/org/checkpoint_org_5309_One_HSN.hdf5')


## Create classifier wrapper
classifier = KerasClassifier(model=model, clip_values=(min_, max_))


logger.info("Create PixelAttack attack")
NeuralNetworkMixin.channels_first = True
adv_crafter = PixelAttackPosition(classifier, th=1)
traintic = time.time()
logger.info("Craft attack on training examples")
XTRAIN["image"][0] = np.squeeze(XTRAIN["image"][0])                      # 删去维度是1的维度
x_train_adv = adv_crafter.generate(XTRAIN, testFile=testFile, dataset=dataset)
traintoc = time.time()
traintick_tock = traintic - traintoc
logger.info('advtraintime{:.4f} (sec)'.format(traintick_tock))


testtic = time.time()
logger.info("Craft attack test examples")
XTEST["image"][0] = np.squeeze(XTEST["image"][0])                        # 删去维度是1的维度
x_test_adv = adv_crafter.generate(XTEST, testFile=testFile, dataset=dataset)
testtoc = time.time()
testtick_tock = testtic - testtoc
logger.info('advtesttime{:.4f} (sec)'.format(testtick_tock))
outputs = calculate_image(model,adv=True,testFile=testFile)


x_train_adv = np.expand_dims(x_train_adv, 1)
x_test_adv = np.expand_dims(x_test_adv, 1)


## Evaluate the classifier on the adversarial samples
classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(x_test_adv,y_test,model)
classification = str(classification)
each_acc = str(each_acc)
logger.info("Classifier before adversarial training")
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
plt.savefig("./result/adv/" + str(testFile) + "/confusion_matrix_adv_" + str(testFile) + "_One_1_HSN_position_"
            + str(dataset) + ".png")
# plt.show()
plt.close()


## Data augmentation: expand the training set with the adversarial samples
x_train = np.append(X_train, x_train_adv, axis=0)
y_train = np.append(y_train, y_train, axis=0)


S = windowSize
L = numPCAcomponents
output_units =  classes
## input layer
input_layer = Input((1, L, S, S))
conv_layer1 = Conv3D(filters=8, kernel_size=(7, 3, 3), activation='relu')(input_layer)
conv_layer2 = Conv3D(filters=16, kernel_size=(5, 3, 3), activation='relu')(conv_layer1)
conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)
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
advmodel = Model(inputs=input_layer, outputs=output_layer)


## Retrain the CNN on the extended dataset
keras_callbacks = [ModelCheckpoint(filepath="checkpoint/adv/checkpoint_adv_" + str(testFile) + "_One_1_HSN_position_"
                   + str(dataset) + ".hdf5", verbose=1, save_best_only=True),
                  ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=25, min_lr=0.000001, verbose=1),
                  CSVLogger("result/adv/" + str(testFile) + "/logs_adv_" + str(testFile) + "_One_1_HSN_position_" + 
                  str(dataset) + ".csv", separator=';', append=True)]  
adam = adam_v2.Adam(lr=0.001, decay=1e-06)                             
advmodel.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

## Start to train model 
advhistory = advmodel.fit(x_train, y_train, 
                    batch_size=256, 
                    epochs=100, 
                    verbose=1, 
                    validation_data=(X_test, y_test),
                    callbacks=keras_callbacks,
                    shuffle=True)
# summarize history for accuracy
plt.plot(advhistory.history['accuracy'])
plt.plot(advhistory.history['val_accuracy'])
plt.title('adversarial trained model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid(True)
plt.legend(['train', 'test'], loc='upper left') 
plt.savefig("./result/adv/" + str(testFile) + "/model_advaccuracy_" + str(testFile) + "_One_1_HSN_position_" 
            + str(dataset) + ".png")
# plt.show()
plt.close()  # 关闭窗口
# summarize history for loss 
plt.plot(advhistory.history['loss'])
plt.plot(advhistory.history['val_loss'])
plt.title('adversarial trained model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.grid(True)
plt.legend(['train', 'test'], loc='upper left') 
plt.savefig("./result/adv/" + str(testFile) + "/model_advloss_" + str(testFile) + "_One_1_HSN_position_" 
            + str(dataset) + ".png")
# plt.show()
plt.close()  # 关闭窗口


## Evaluate the adversarially trained classifier on the test set
Adv_classification, Adv_confusion, Adv_Test_loss, Adv_Test_accuracy, oa, each_acc, aa, kappa = reports(X_test,y_test,advmodel)
Adv_classification = str(Adv_classification)
each_acc = str(each_acc)
logger.info("Classifier after adversarial training")
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
            "_One_1_HSN_position_" + str(dataset) + ".png")
# plt.show()
plt.close()
outputs = calculate_image(advmodel,testFile=testFile)
spectral.save_rgb("test_result/adv/" + str(testFile) + "/predictions_after_adv_" + str(testFile) + 
            "_One_1_HSN_position_" + str(dataset) + ".png", outputs, colors=spectral.spy_colors)
logger.info("Finish!")