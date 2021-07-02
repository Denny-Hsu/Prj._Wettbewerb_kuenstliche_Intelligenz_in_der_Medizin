# from tqdm import tqdm
# import csv
# import scipy.io as sio
# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
# import os
# from os import listdir
# from os.path import isfile, join

# import pandas as pd
# import collections
# import cv2


# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
# from sklearn.neighbors import KNeighborsClassifier

# from sklearn.metrics import confusion_matrix, accuracy_score,f1_score,recall_score,precision_score
# from ecgdetectors import Detectors

# # import keras
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, MaxPool2D, GlobalAveragePooling2D, Flatten, BatchNormalization
# from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
# # from keras.utils import plot_model
# # from keras.optimizers import SGD
# # from keras import regularizers
# # from keras.callbacks import EarlyStopping
# # from keras.callbacks import ModelCheckpoint
# # from keras.callbacks import ReduceLROnPlateau
# # from keras.utils.np_utils import to_categorical

# from tensorflow.keras import backend as K
# from sklearn.metrics import fbeta_score


# from numpy import mean
# from sklearn.datasets import make_classification
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.tree import DecisionTreeClassifier
# from imblearn.pipeline import Pipeline

# from imblearn.over_sampling import RandomOverSampler, SMOTE
# # from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, KMeansSMOTE
# from imblearn.under_sampling import RandomUnderSampler

# from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
# from wettbewerb import load_references

# ecg_leads,ecg_labels,fs,ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz

# dataset = ecg_leads
# if os.path.exists("../training/ecg_images"):
#     print("File exists.")
# else:
#     os.mkdir("../training/ecg_images")

# for i in tqdm(range(len(dataset))):
#     data = ecg_leads[i].reshape(len(ecg_leads[i]),1)

#     plt.figure(figsize=(60, 5))
#     plt.xlim(0,len(ecg_leads[i]))
#     plt.plot(data, color='black', linewidth = 0.1)
#     plt.savefig('../training/ecg_images/{}.png'.format(ecg_names[i]))

#     plt.close()

# onlyfiles = [f for f in listdir("../training/ecg_images") if isfile(join("../training//ecg_images", f))]

# df = pd.read_csv("../training/REFERENCE.csv", header=None)
# x = []
# y = []
# for i in range(len(onlyfiles)):
#     if (df.iloc[i][1] == "N") or (df.iloc[i][1] == "A"):
#         image = cv2.imread("../training/ecg_images/{}".format(onlyfiles[i]))
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         resize_x = cv2.resize(gray, (128, 1024))
#         reshape_x = np.asarray(resize_x).reshape(resize_x.shape[0], resize_x.shape[1], 1)
#         x.append(reshape_x)
#         y.append(df.iloc[i][1])

# for n, i in enumerate(y):
#     if i == "N":
#         y[n] = 0
#     elif i == "A":
#         y[n] = 1

# x = np.asarray(x).astype(int)
# y = np.asarray(y).astype(int)

# train_images, train_labels = x, y
# # # Normalize pixel values to be between 0 and 1
# train_images = train_images / 255.0

# #####
# THRESHOLD = 0.5
# def precision(y_true, y_pred, threshold_shift=0.5-THRESHOLD):

#     # just in case
#     y_pred = K.clip(y_pred, 0, 1)

#     # shifting the prediction threshold from .5 if needed
#     y_pred_bin = K.round(y_pred + threshold_shift)

#     tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
#     fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))

#     precision = tp / (tp + fp)
#     return precision


# def recall(y_true, y_pred, threshold_shift=0.5-THRESHOLD):

#     # just in case
#     y_pred = K.clip(y_pred, 0, 1)

#     # shifting the prediction threshold from .5 if needed
#     y_pred_bin = K.round(y_pred + threshold_shift)

#     tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
#     fn = K.sum(K.round(K.clip(y_true - y_pred_bin, 0, 1)))

#     recall = tp / (tp + fn)
#     return recall

# def fbeta(y_true, y_pred, threshold_shift=0.5-THRESHOLD):
#     beta = 1

#     # just in case
#     y_pred = K.clip(y_pred, 0, 1)

#     # shifting the prediction threshold from .5 if needed
#     y_pred_bin = K.round(y_pred + threshold_shift)

#     tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
#     fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
#     fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)

#     beta_squared = beta ** 2
#     return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall)
# #####

# model = Sequential()

# model.add(Conv2D(128, (3,3), activation='relu', input_shape = (1024, 128, 1)))
# model.add(BatchNormalization())
# model.add(MaxPool2D(2,2))
# # model.add(Dropout(0.2))

# model.add(Conv2D(128, (3,3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPool2D(2,2))
# model.add(Dropout(0.5))

# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPool2D(2,2))
# model.add(Dropout(0.5))

# model.add(Conv2D(32, (3,3), activation='relu'))
# model.add(Conv2D(32, (3,3), activation='relu'))

# model.add(BatchNormalization())
# # model.add(MaxPool2D(2,2))
# model.add(Dropout(0.5))

# model.add(Flatten())


# model.add(Dense(256, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Dense(128, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Dense(64, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Dense(32, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Dense(2, activation='softmax'))

# model.summary()

# model.compile(optimizer='adam',   # change: use SGD
#               loss='sparse_categorical_crossentropy',
#               metrics=[fbeta, 'accuracy'])

# history = model.fit(train_images, train_labels, batch_size=128, epochs = 500, validation_split=0.25, verbose = 1)  #epochs = 500


# model.save('./pred_model.h5')

# ====== START ======

''' Gradient Boosting Classifier '''
# import csv
# import scipy.io as sio
# import matplotlib.pyplot as plt
# import numpy as np
# from ecgdetectors import Detectors
# import os
#
# import pandas as pd
# import neurokit2 as nk
# from sklearn.datasets import make_hastie_10_2
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
# from wettbewerb import load_references
# from imblearn.over_sampling import SMOTE
#
# import warnings
# warnings.filterwarnings("ignore")
#
#
# ecg_leads, ecg_labels, fs, ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz
#
# features = np.array([])
# detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors
# labels = np.array([])
#
# for idx, ecg_lead in enumerate(ecg_leads):
#     if (ecg_labels[idx] == "N") or (ecg_labels[idx] == "A"):
#         peaks, info = nk.ecg_peaks(ecg_lead, sampling_rate= 200)
#         peaks = peaks.astype('float64')
#         hrv = nk.hrv_time(peaks, sampling_rate= fs)
#         hrv = hrv.astype('float64')
#         features = np.append(features, [hrv['HRV_CVNN'], hrv['HRV_CVSD'], hrv['HRV_HTI'], hrv['HRV_IQRNN'], hrv['HRV_MCVNN'], hrv['HRV_MadNN'],  hrv['HRV_MeanNN'], hrv['HRV_MedianNN'], hrv['HRV_RMSSD'], hrv['HRV_SDNN'], hrv['HRV_SDSD'], hrv['HRV_TINN'], hrv['HRV_pNN20'],hrv['HRV_pNN50'] ])
#         features = features.astype('float64')
#         labels = np.append(labels, ecg_labels[idx])
#
# features= features.reshape(int(len(features)/14), 14)
# x = np.isnan(features)
# # replacing NaN values with 0
# features[x] = 0
#
# X_train = features
# y_train = labels
#
# x_over, y_over = SMOTE(random_state=42).fit_resample(X_train, y_train)
#
# clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0).fit(x_over , y_over)
#
# import joblib
#
# joblib.dump(clf, 'GradientBoostingClassifier')
# print('Finished for training and saved model.')

# from sklearn.metrics import classification_report
# print(classification_report(y_test, clf_over_pred))
#----------------------------------------
''' Convolutional Neural Network '''

# import csv
# import scipy.io as sio
# import matplotlib.pyplot as plt
# import numpy as np
# from ecgdetectors import Detectors
# import os
#
# import pandas as pd
# import neurokit2 as nk
# from sklearn.datasets import make_hastie_10_2
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
#
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.optimizers import SGD, Adam
# from tensorflow.keras import backend as K
# from keras.utils.np_utils import *
# import numpy as np
#
#
# import warnings
# warnings.filterwarnings("ignore")
#
# ecg_leads, ecg_labels, fs, ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz
#
# features = np.array([])
# detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors
# labels = np.array([])
#
# for idx, ecg_lead in enumerate(ecg_leads):
#     if (ecg_labels[idx] == "N") or (ecg_labels[idx] == "A"):
#         peaks, info = nk.ecg_peaks(ecg_lead, sampling_rate=fs)
#         hrv = nk.hrv_time(peaks, sampling_rate=fs)
#         features = np.append(features, [hrv['HRV_CVNN'], hrv['HRV_CVSD'], hrv['HRV_HTI'], hrv['HRV_IQRNN'], hrv['HRV_MCVNN'], hrv['HRV_MadNN'],  hrv['HRV_MeanNN'], hrv['HRV_MedianNN'], hrv['HRV_RMSSD'], hrv['HRV_SDNN'], hrv['HRV_SDSD'], hrv['HRV_TINN'], hrv['HRV_pNN20'],hrv['HRV_pNN50'] ])
#         labels = np.append(labels, ecg_labels[idx])
#
# features= features.reshape(int(len(features)/14), 14)
# x = np.isnan(features)
# # replacing NaN values with 0
# features[x] = 0
# features = features.astype('float64')
#
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=10,stratify=labels)
#
# from imblearn.over_sampling import SMOTE
#
# x_over, y_over = SMOTE(random_state=42).fit_resample(X_train, y_train)
#
# X_train_arr = np.array(x_over).reshape(np.array(x_over).shape[0], np.array(x_over).shape[1], 1)
# X_test_arr = np.array(X_test).reshape(np.array(X_test).shape[0], np.array(X_test).shape[1], 1)
#
# for n, i in enumerate(y_test):
#     if i == "N":
#         y_test[n] = 0
#     elif i == "A":
#         y_test[n] = 1
# for n, i in enumerate(y_over):
#     if i == "N":
#         y_over[n] = 0
#     elif i == "A":
#         y_over[n] = 1
#
# y_train_categorical = to_categorical(y_over, num_classes=2)
# y_test_categorical = to_categorical(y_test, num_classes=2)
#
# ### Functions
# def f1(y_true, y_pred):
#     def recall(y_true, y_pred):
#         """Recall metric.
#
#         Only computes a batch-wise average of recall.
#
#         Computes the recall, a metric for multi-label classification of
#         how many relevant items are selected.
#         """
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#         recall = true_positives / (possible_positives + K.epsilon())
#         return recall
#
#     def precision(y_true, y_pred):
#         """Precision metric.
#
#         Only computes a batch-wise average of precision.
#
#         Computes the precision, a metric for multi-label classification of
#         how many selected items are relevant.
#         """
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#         precision = true_positives / (predicted_positives + K.epsilon())
#         return precision
#     precision = precision(y_true, y_pred)
#     recall = recall(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))
#
#
# def f1_weighted(true, pred):  # shapes (batch, 4)
#
#     # for metrics include these two lines, for loss, don't include them
#     # these are meant to round 'pred' to exactly zeros and ones
#     # predLabels = K.argmax(pred, axis=-1)
#     # pred = K.one_hot(predLabels, 4)
#
#     ground_positives = K.sum(true, axis=0) + K.epsilon()  # = TP + FN
#     pred_positives = K.sum(pred, axis=0) + K.epsilon()  # = TP + FP
#     true_positives = K.sum(true * pred, axis=0) + K.epsilon()  # = TP
#     # all with shape (4,)
#
#     precision = true_positives / pred_positives
#     recall = true_positives / ground_positives
#     # both = 1 if ground_positives == 0 or pred_positives == 0
#     # shape (4,)
#
#     f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
#     # still with shape (4,)
#
#     weighted_f1 = f1 * ground_positives / K.sum(ground_positives)
#     weighted_f1 = K.sum(weighted_f1)
#
#     return 1 - weighted_f1  # for metrics, return only 'weighted_f1'
#
# ###
#
# #original
# model = Sequential()
#
# model.add(Conv1D(8, 3, activation='relu', input_shape=(14,1)))
# model.add(BatchNormalization())
# model.add(MaxPooling1D(1))
#
# model.add(Conv1D(8, 3, activation='relu'))
# model.add(BatchNormalization())
#
# model.add(Conv1D(8, 3, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.1))
#
# model.add(Conv1D(8, 3, activation='relu'))
# model.add(BatchNormalization())
#
# model.add(Conv1D(8, 3, activation='relu'))
# model.add(BatchNormalization())
#
# model.add(Flatten())
# model.add(Dense(32, activation = 'sigmoid'))
# model.add(Dense(2, activation = 'softmax'))
#
# model.summary()
#
# from keras.callbacks import ModelCheckpoint
#
# adam = Adam(learning_rate=0.001, decay=1e-6)
# sgd = SGD(lr=0.00005, decay=1e-6, momentum=0.9, nesterov=True)
# CheckPoint = ModelCheckpoint('best_model.h5',  # model filename
#                              monitor='f1', # quantity to monitor
#                              verbose= 1, # verbosity - 0 or 1
#                              save_best_only= True, # The latest best model will not be overwritten
#                              mode='max')
#
# model.compile(optimizer=sgd,   # change: use SGD
#               loss=f1_weighted, #'binary_crossentropy' #'mean_squared_error' #categorical_crossentropy
#               metrics=[f1,"binary_accuracy"])
#
# history = model.fit(X_train_arr, y_train_categorical, epochs=500, verbose=1, callbacks=[CheckPoint], validation_data=(X_test_arr, y_test_categorical))
#
# test_loss, test_f1, test_acc = model.evaluate(X_test_arr, y_test_categorical, verbose=2)
# print('Accuracy',test_acc)
# print('F1 score',test_f1)
#
# from tensorflow.keras.models import load_model
#
# cnn_best_model = load_model('cnn_best_model.h5', custom_objects={'f1_weighted': f1_weighted, 'f1': f1})
#
# cnn_best_model.compile(optimizer=sgd,   # change: use SGD
#               loss=f1_weighted, #'binary_crossentropy' #'mean_squared_error' #categorical_crossentropy
#               metrics=[f1,"binary_accuracy"])
#
# #  Evaluate the with the loaded model
# test_loss, test_f1, test_acc = cnn_best_model.evaluate(X_test_arr, y_test_categorical, verbose=2)
# print('Accuracy',test_acc)
# print('F1 score',test_f1)
#----------------------------------------
''' AdaBoost Classifier '''

# import csv
# import scipy.io as sio
# import matplotlib.pyplot as plt
# import numpy as np
# from ecgdetectors import Detectors
# import os
#
# import pandas as pd
# import neurokit2 as nk
# from sklearn.datasets import make_hastie_10_2
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
# from wettbewerb import load_references
# from imblearn.over_sampling import SMOTE
#
# import warnings
# warnings.filterwarnings("ignore")
#
#
# ecg_leads, ecg_labels, fs, ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz
#
# features = np.array([])
# detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors
# labels = np.array([])
#
# for idx, ecg_lead in enumerate(ecg_leads):
#     if (ecg_labels[idx] == "N") or (ecg_labels[idx] == "A"):
#         peaks, info = nk.ecg_peaks(ecg_lead, sampling_rate= fs)
#         peaks = peaks.astype('float64')
#         hrv = nk.hrv_time(peaks, sampling_rate= fs)
#         hrv = hrv.astype('float64')
#         features = np.append(features, [hrv['HRV_CVNN'], hrv['HRV_CVSD'], hrv['HRV_HTI'], hrv['HRV_IQRNN'], hrv['HRV_MCVNN'], hrv['HRV_MadNN'],  hrv['HRV_MeanNN'], hrv['HRV_MedianNN'], hrv['HRV_RMSSD'], hrv['HRV_SDNN'], hrv['HRV_SDSD'], hrv['HRV_TINN'], hrv['HRV_pNN20'],hrv['HRV_pNN50'] ])
#         features = features.astype('float64')
#         labels = np.append(labels, ecg_labels[idx])
#
# features= features.reshape(int(len(features)/14), 14)
# x = np.isnan(features)
# # replacing NaN values with 0
# features[x] = 0
#
# X_train = features
# y_train = labels
#
# x_over, y_over = SMOTE(random_state=42).fit_resample(X_train, y_train)
#
# ada = AdaBoostClassifier(n_estimators=150, learning_rate=0.4).fit(x_over , y_over)
#
# import joblib
#
# joblib.dump(ada, 'AdaBoostClassifier')
# print('Finished for training and saved model.')

# from sklearn.metrics import classification_report
# print(classification_report(y_test, clf_over_pred))
#----------------------------------------
''' XGBoost Classifier '''

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os

import pandas as pd
import neurokit2 as nk
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
from wettbewerb import load_references
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")


ecg_leads, ecg_labels, fs, ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz

features = np.array([])
detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors
labels = np.array([])

for idx, ecg_lead in enumerate(ecg_leads):
    if (ecg_labels[idx] == "N") or (ecg_labels[idx] == "A"):
        peaks, info = nk.ecg_peaks(ecg_lead, sampling_rate= fs)
        peaks = peaks.astype('float64')
        hrv = nk.hrv_time(peaks, sampling_rate= fs)
        hrv = hrv.astype('float64')
        features = np.append(features, [hrv['HRV_CVNN'], hrv['HRV_CVSD'], hrv['HRV_HTI'], hrv['HRV_IQRNN'], hrv['HRV_MCVNN'], hrv['HRV_MadNN'],  hrv['HRV_MeanNN'], hrv['HRV_MedianNN'], hrv['HRV_RMSSD'], hrv['HRV_SDNN'], hrv['HRV_SDSD'], hrv['HRV_TINN'], hrv['HRV_pNN20'],hrv['HRV_pNN50'] ])
        features = features.astype('float64')
        labels = np.append(labels, ecg_labels[idx])

features= features.reshape(int(len(features)/14), 14)
x = np.isnan(features)
# replacing NaN values with 0
features[x] = 0

X_train = features
y_train = labels

x_over, y_over = SMOTE(random_state=42).fit_resample(X_train, y_train)

import xgboost as xgb

xgb_C = xgb.XGBClassifier(objective='reg:linear', colsample_bytree = 0.3, learning_rate = 0.55,
                max_depth = 5, alpha = 3, n_estimators = 150).fit(x_over, y_over)


import joblib

joblib.dump(xgb_C, 'XGBoostClassifier')
print('Finished for training and saved model.')

# from sklearn.metrics import classification_report
# print(classification_report(y_test, clf_over_pred))