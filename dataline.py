# Import the necessary libraries
import numpy as np
import cv2
from sklearn.decomposition import PCA
import scipy.io as sio
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn import preprocessing
import os
import random
from random import shuffle
from skimage.transform import rotate
import scipy.ndimage
import spectral 

img = spectral.open_image('HSI_data/92AV3C.lan')
img_1 = img[:,:,19].reshape(145,145)
view = spectral.imshow(img)
cv2.imshow('1',img)
cv2.waitKey(0)