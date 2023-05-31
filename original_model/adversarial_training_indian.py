# -*- coding: utf-8 -*-
"""
Trains a convolutional neural network on the Indian_pines dataset, then generated adversarial images using the
DeepFool attack and retrains the network on the training set augmented with the adversarial images.
"""
## Configure
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import h5py
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
from keras.optimizer_v2.gradient_descent import SGD
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_first')
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

from art.attacks.evasion import DeepFool
from art.attacks.evasion.pixel_threshold import PixelAttack
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.estimators.classification import KerasClassifier
from art.utils import load_dataset

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# sess = tf.compat.v1.Session(config=config)


## Configure a logger to capture ART outputs; these are printed in console and the level of detail is set to INFO
# 配置记录器来捕获ART输出;这些信息会在控制台中打印出来，详细信息的级别设置为INFO
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Global Variables
# The number of principal components to be retained in the PCA algorithm, 
# the number of retained features  n
numPCAcomponents = 30
# Patches windows size
windowSize = 5
# The proportion of Test sets
testRatio = 0.90

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

# Reshape data into (numberofsumples, channels, height, width)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[3], 
                               X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[3], 
                             X_test.shape[1], X_test.shape[2]))

# convert class labels to on-hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

min_ = np.min(X_train)
max_ = np.max(X_train)

# Define the input shape 
input_shape= X_train[0].shape
print(input_shape)

# number of filters
C1 = 3*numPCAcomponents


## Define the model structure
model = Sequential()

model.add(Conv2D(C1, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(3*C1, (3, 3), activation='relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(6*numPCAcomponents, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='softmax'))


## Define optimization and train method
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=25, 
                              min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath="checkpoint.hdf5", verbose=1, 
                              save_best_only=True)
sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


## Create classifier wrapper
classifier = KerasClassifier(model=model, clip_values=(min_, max_))


## Start to train model 
history = model.fit(X_train, y_train, 
                    batch_size=256, 
                    epochs=1, 
                    verbose=1, 
                    validation_data=(X_test, y_test),
                    callbacks=[reduce_lr, checkpointer],
                    shuffle=True)
# model.save('./model/HSI_model_epochs100041701.h5')


## using plot_model module to save the model figure
# plot_model(model, to_file='./model/model.png', show_shapes=True)
# print(history.history.keys())


## show the model figure
# model_img = plt.imread('./model/model.png')
# plt.imshow(model_img)
# plt.show()
# print('val_acc before adv = {:.4f}'.format(np.max(history.history['val_accuracy'])))

# summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.grid(True)
# plt.legend(['train', 'test'], loc='upper left') 
# plt.savefig("./result/org/model_accuracy_100041701_One.png")
# plt.show()

# summarize history for loss 
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.grid(True)
# plt.legend(['train', 'test'], loc='upper left') 
# plt.savefig("./result/org/model_loss_100041701_One.png")
# plt.show()


## Craft adversarial samples with DeepFool
# logger.info("Create DeepFool attack")
# adv_crafter = DeepFool(classifier)
# logger.info("Create PGD attack")
# adv_crafter = ProjectedGradientDescent(classifier, eps=8 / 255, eps_step=2 / 255, max_iter=10, num_random_init=1)
# logger.info("Create FGSM attack")
# adv_crafter = FastGradientMethod(classifier, eps=8/255, eps_step=2/255)

logger.info("Create PixelAttack attack")
adv_crafter = PixelAttack(classifier, th=1)

logger.info("Craft attack on training examples")
x_train_adv = adv_crafter.generate(X_train)
logger.info("Craft attack test examples")
x_test_adv = adv_crafter.generate(X_test)


## Evaluate the classifier on the adversarial samples
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info("Classifier before adversarial training")
logger.info("Accuracy on adversarial samples: %.2f%%", (acc * 100))


## Data augmentation: expand the training set with the adversarial samples
x_train = np.append(X_train, x_train_adv, axis=0)
y_train = np.append(y_train, y_train, axis=0)


## Retrain the CNN on the extended dataset
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
advcheckpointer = ModelCheckpoint(filepath="checkpoint_adv041701_One.hdf5", verbose=1, 
                              save_best_only=True)
advhistory = model.fit(x_train, y_train, 
                    batch_size=32, 
                    epochs=100, 
                    verbose=1, 
                    validation_data=(X_test, y_test),
                    callbacks=[reduce_lr, advcheckpointer],
                    shuffle=True)

print('val_acc after adv = {.4f}'.format(np.max(advhistory.history['val_accuracy'])))
# summarize history for accuracy
plt.plot(advhistory.history['accuracy'])
plt.plot(advhistory.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid(True)
plt.legend(['train', 'test'], loc='upper left') 
plt.savefig("./result/adv/model_advaccuracy_100041701_One.png")
plt.show()

# summarize history for loss 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.grid(True)
plt.legend(['train', 'test'], loc='upper left') 
plt.savefig("./result/adv/model_advloss_100041701_One.png")
plt.show()

## Evaluate the adversarially trained classifier on the test set
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info("Classifier with adversarial training")
logger.info("Accuracy on adversarial samples: %.2f%%", (acc * 100))


orgmodel = load_model('checkpoint_org041701_One.hdf5')
orgloss,orgacc = orgmodel.evaluate(X_test,y_test,batch_size=32,verbose=0)
advmodel = load_model('checkpoint_adv041701_One.hdf5')
advloss,advacc = advmodel.evaluate(X_test,y_test,batch_size=32,verbose=0)
print('orgtest_loss:',orgloss,'orgtest_accuracy:',orgacc,'advtest_loss:',advloss,'advtest_accuracy:',advacc)