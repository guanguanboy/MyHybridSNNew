import keras
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization
from keras.layers import Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

from operator import truediv

from plotly.offline import init_notebook_mode

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import spectral

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

## GLOBAL VARIABLES
dataset = 'IP'
test_ratio = 0.7
windowSize = 25

def loadData(name):
    data_path = os.path.join(os.getcwd(), 'data')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']

    return data, labels


X, y = loadData(dataset)

print(X.shape, y.shape)

view = spectral.imshow(X, (29, 19, 9))
#view = spectral.imshow(X, (1, 2, 3))

print(view)
plt.show()

#show grandtruth
gt_view = spectral.imshow(classes=y)
plt.show()

#show both the data and the class value
img = X
gt = y

view = spectral.imshow(img, (30, 20, 10), classes=gt)
view.set_display_mode('overlay')
view.class_alpha = 0.5
plt.show()

#saving rgb image files
spectral.save_rgb('rgb.jpg', img, [29, 19, 9])

spectral.save_rgb('gt.jpg', gt, colors=spectral.spy_colors)

#spectral.view_cube(img, spectral.bands=[29, 19, 9])

spectral.view_cube(img,bands=[29, 19, 9])
plt.show()

