from tqdm import tqdm
import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import collections
import cv2
from os import listdir
from os.path import isfile, join

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, accuracy_score,f1_score,recall_score,precision_score
from ecgdetectors import Detectors

import keras
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten
from keras.layers import Dense, Dropout, LeakyReLU
# from keras.utils import plot_model
from keras.optimizers import SGD
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.utils.np_utils import to_categorical

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import fbeta_score


from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline

from imblearn.over_sampling import RandomOverSampler, SMOTE
# from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler

from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from wettbewerb import load_references

ecg_leads,ecg_labels,fs,ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz

dataset = ecg_leads
for i in tqdm(range(len(dataset))):
    data = ecg_leads[i].reshape(len(ecg_leads[i]),1)
    plt.figure(figsize=(60, 5))
    plt.xlim(0,len(ecg_leads[i]))
    plt.plot(data, color='black', linewidth = 0.1)
    plt.savefig('../ecg_images/{}.png'.format(ecg_names[i]))


onlyfiles = [f for f in listdir("./ecg_images") if isfile(join("./ecg_images", f))]

df = pd.read_csv("./training/REFERENCE.csv", header=None)
x = []
y = []
for i in range(len(onlyfiles)):
    if (df.iloc[i][1] == "N") or (df.iloc[i][1] == "A"):
        image = cv2.imread("./ecg_images/{}".format(onlyfiles[i]))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resize_x = cv2.resize(gray, (128, 1024))
        reshape_x = np.asarray(resize_x).reshape(resize_x.shape[0], resize_x.shape[1], 1)
        x.append(reshape_x)
        y.append(df.iloc[i][1])

for n, i in enumerate(y):
    if i == "N":
        y[n] = 0
    elif i == "A":
        y[n] = 1

x = np.asarray(x).astype(int)
y = np.asarray(y).astype(int)

train_images, train_labels = x, y
# # Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0

#####
THRESHOLD = 0.5
def precision(y_true, y_pred, threshold_shift=0.5-THRESHOLD):

    # just in case
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))

    precision = tp / (tp + fp)
    return precision


def recall(y_true, y_pred, threshold_shift=0.5-THRESHOLD):

    # just in case
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fn = K.sum(K.round(K.clip(y_true - y_pred_bin, 0, 1)))

    recall = tp / (tp + fn)
    return recall

def fbeta(y_true, y_pred, threshold_shift=0.5-THRESHOLD):
    beta = 1

    # just in case
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall)
#####

model = models.Sequential()

model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape = (360, 4320, 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(2,2))
# model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(2,2))
# model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(2,2))
# model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(2,2))
# model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(2,2))
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(2,2))
model.add(layers.Dropout(0.5))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))


model.add(layers.Dense(2, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',   # change: use SGD
              loss='sparse_categorical_crossentropy',
              metrics=[fbeta, 'accuracy'])

history = model.fit(train_images, train_labels, batch_size=128, epochs = 500, validation_split=0.25)  #epochs = 500


model.save('pred_model.h5')
