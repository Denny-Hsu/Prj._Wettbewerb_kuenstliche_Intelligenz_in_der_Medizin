# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy.stats
from imblearn.over_sampling import SMOTE
import pickle

from collections import Counter
from ecgdetectors import Detectors
import os
import pandas as pd
from wettbewerb import load_references
import joblib
import neurokit2 as nk


###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads,fs,ecg_names,use_pretrained=False):
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.

    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''
#------------------------------------------------------------------------------
# Euer Code ab hier
#     model_name = "model.npy"
#     if use_pretrained:
#         model_name = "model_pretrained.npy"
#     with open(model_name, 'rb') as f:
#         th_opt = np.load(f)         # Lade simples Model (1 Parameter)
#
#     detectors = Detectors(fs)        # Initialisierung des QRS-Detektors
#
#     predictions = list()
#
#     for idx,ecg_lead in enumerate(ecg_leads):
#         r_peaks = detectors.hamilton_detector(ecg_lead)     # Detektion der QRS-Komplexe
#         sdnn = np.std(np.diff(r_peaks)/fs*1000)
#         if sdnn < th_opt:
#             predictions.append((ecg_names[idx], 'N'))
#         else:
#             predictions.append((ecg_names[idx], 'A'))
#         if ((idx+1) % 100)==0:
#             print(str(idx+1) + "\t Dateien wurden verarbeitet.")
#
#
# #------------------------------------------------------------------------------
#     return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!

    # Load test-dataset
#     from tqdm import tqdm
#     import os
#     from os import listdir
#     from os.path import isfile, join
#     import cv2

    #####
#     THRESHOLD = 0.5

#     def precision(y_true, y_pred, threshold_shift=0.5 - THRESHOLD):

#         # just in case
#         y_pred = K.clip(y_pred, 0, 1)

#         # shifting the prediction threshold from .5 if needed
#         y_pred_bin = K.round(y_pred + threshold_shift)

#         tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
#         fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))

#         precision = tp / (tp + fp)
#         return precision

#     def recall(y_true, y_pred, threshold_shift=0.5 - THRESHOLD):

#         # just in case
#         y_pred = K.clip(y_pred, 0, 1)

#         # shifting the prediction threshold from .5 if needed
#         y_pred_bin = K.round(y_pred + threshold_shift)

#         tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
#         fn = K.sum(K.round(K.clip(y_true - y_pred_bin, 0, 1)))

#         recall = tp / (tp + fn)
#         return recall

#     def fbeta(y_true, y_pred, threshold_shift=0.5 - THRESHOLD):
#         beta = 1

#         # just in case
#         y_pred = K.clip(y_pred, 0, 1)

#         # shifting the prediction threshold from .5 if needed
#         y_pred_bin = K.round(y_pred + threshold_shift)

#         tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
#         fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
#         fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

#         precision = tp / (tp + fp)
#         recall = tp / (tp + fn)

#         beta_squared = beta ** 2
#         return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall)

    #####

    # ecg_leads,ecg_labels,fs,ecg_names = load_references('../test/') # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz

#     test_set = ecg_leads
#     if os.path.exists("../test/ecg_images"):
#         print("File exists.")
#     else:
#         os.mkdir("../test/ecg_images")
#     for i in tqdm(range(len(test_set))):
#         data = ecg_leads[i].reshape(len(ecg_leads[i]), 1)
#         plt.figure(figsize=(60, 5))
#         plt.xlim(0, len(ecg_leads[i]))
#         # plt.plot(data, color='black', linewidth=0.1)
#         plt.savefig('../test/ecg_images/{}.png'.format(ecg_names[i]))

#     onlyfiles = [f for f in listdir("../test/ecg_images") if isfile(join("../test/ecg_images", f))]

#     df = pd.read_csv("../test/REFERENCE.csv", header=None)
#     x = []
#     y = []
#     name_test =[]

#     for i in range(len(onlyfiles)):
#         if (df.iloc[i][1] == "N") or (df.iloc[i][1] == "A"):
#             image = cv2.imread("../test/ecg_images/{}".format(onlyfiles[i]))
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             resize_x = cv2.resize(gray, (128, 1024))
#             reshape_x = np.asarray(resize_x).reshape(resize_x.shape[0], resize_x.shape[1], 1)
#             x.append(reshape_x)
#             y.append(df.iloc[i][1])
#             name_test.append(df.iloc[i])


#     for n, i in enumerate(y):
#         if i == "N":
#             y[n] = 0
#         elif i == "A":
#             y[n] = 1

#     x = np.asarray(x).astype(int)
#     y = np.asarray(y).astype(int)

#     test_images, test_labels = x, y
    # # Normalize pixel values to be between 0 and 1
#     test_images = test_images / 255.0

#     import keras
#     import tensorflow as tf
#     from keras.models import load_model

    # model = load_model('./pred_model.h5', custom_objects={'fbeta': fbeta})
#     model = tf.keras.models.load_model('./pred_model.h5', custom_objects={'fbeta': fbeta})
    # model = tf.keras.models.load_model('./pred_model.h5', custom_objects={'fbeta': fbeta})
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)  # .from_saved_model(saved_model_dir)
    # tflite_model = converter.convert()
    # open("model.tflite", "wb").write(tflite_model)

#     pred_labels = model.predict_classes(test_images)
    
#     pred_labels = np.asarray(pred_labels).astype('str')
#     for n, i in enumerate(pred_labels):
#         if i == '0':
#             pred_labels[n] = "N"
#         elif i == '1':
#             pred_labels[n] = "A"
    #-------------------------------------------------------------------------------
    ''' Gradient Boosting Classfier '''
    # import warnings
    # warnings.filterwarnings("ignore")
    #
    # test_features = np.array([])
    #
    # for idx, ecg_lead in enumerate(ecg_leads):
    #     peaks, info = nk.ecg_peaks(ecg_lead, sampling_rate= 200)
    #     peaks = peaks.astype('float64')
    #     hrv = nk.hrv_time(peaks, sampling_rate= fs)
    #     hrv = hrv.astype('float64')
    #     test_features = np.append(test_features, [hrv['HRV_CVNN'], hrv['HRV_CVSD'], hrv['HRV_HTI'], hrv['HRV_IQRNN'], hrv['HRV_MCVNN'], hrv['HRV_MadNN'],  hrv['HRV_MeanNN'], hrv['HRV_MedianNN'], hrv['HRV_RMSSD'], hrv['HRV_SDNN'], hrv['HRV_SDSD'], hrv['HRV_TINN'], hrv['HRV_pNN20'],hrv['HRV_pNN50'] ])
    #     test_features = test_features.astype('float64')
    #
    # test_features= test_features.reshape(int(len(test_features)/14), 14)
    # x = np.isnan(test_features)
    # # replacing NaN values with 0
    # test_features[x] = 0
    #
    # X_test = test_features
    #
    # # with trained model to predict
    # loaded_model = joblib.load('GradientBoostingClassifier')
    # pred_labels = loaded_model.predict(X_test)
    
    # # a list of tuple
    # predictions = list()
    #
    # for idx in range(len(X_test)):
    #     predictions.append((ecg_names[idx], pred_labels[idx]))

    ''' Convolutional Neural Network '''
    import warnings
    warnings.filterwarnings("ignore")
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import SGD
    import numpy as np

    ###
    def f1(y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def f1_weighted(true, pred):  # shapes (batch, 4)

        # for metrics include these two lines, for loss, don't include them
        # these are meant to round 'pred' to exactly zeros and ones
        # predLabels = K.argmax(pred, axis=-1)
        # pred = K.one_hot(predLabels, 4)

        ground_positives = K.sum(true, axis=0) + K.epsilon()  # = TP + FN
        pred_positives = K.sum(pred, axis=0) + K.epsilon()  # = TP + FP
        true_positives = K.sum(true * pred, axis=0) + K.epsilon()  # = TP
        # all with shape (4,)

        precision = true_positives / pred_positives
        recall = true_positives / ground_positives
        # both = 1 if ground_positives == 0 or pred_positives == 0
        # shape (4,)

        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        # still with shape (4,)

        weighted_f1 = f1 * ground_positives / K.sum(ground_positives)
        weighted_f1 = K.sum(weighted_f1)

        return 1 - weighted_f1  # for metrics, return only 'weighted_f1'
    ###

    cnn_best_model = load_model('cnn_best_model.h5', custom_objects={'f1_weighted': f1_weighted, 'f1': f1})

    sgd = SGD(lr=0.00005, decay=1e-6, momentum=0.9, nesterov=True)
    cnn_best_model.compile(optimizer=sgd,  # change: use SGD
                           loss=f1_weighted,  # 'binary_crossentropy' #'mean_squared_error' #categorical_crossentropy
                           metrics=[f1, "binary_accuracy"])

    test_features = np.array([])

    for idx, ecg_lead in enumerate(ecg_leads):
        peaks, info = nk.ecg_peaks(ecg_lead, sampling_rate= 200)
        peaks = peaks.astype('float64')
        hrv = nk.hrv_time(peaks, sampling_rate= fs)
        hrv = hrv.astype('float64')
        test_features = np.append(test_features, [hrv['HRV_CVNN'], hrv['HRV_CVSD'], hrv['HRV_HTI'], hrv['HRV_IQRNN'], hrv['HRV_MCVNN'], hrv['HRV_MadNN'],  hrv['HRV_MeanNN'], hrv['HRV_MedianNN'], hrv['HRV_RMSSD'], hrv['HRV_SDNN'], hrv['HRV_SDSD'], hrv['HRV_TINN'], hrv['HRV_pNN20'],hrv['HRV_pNN50'] ])
        test_features = test_features.astype('float64')

    test_features= test_features.reshape(int(len(test_features)/14), 14)
    x = np.isnan(test_features)
    # replacing NaN values with 0
    test_features[x] = 0

    X_test = test_features
    X_test_arr = np.array(X_test).reshape(np.array(X_test).shape[0], np.array(X_test).shape[1], 1)
    # with trained model to predict
    y_pred = cnn_best_model.predict(X_test_arr)
    pred_labels = [np.argmax(y, axis=None, out=None) for y in y_pred]

    for n, i in enumerate(pred_labels):
        if i == 0:
            pred_labels[n] = 'N'
        if i == 1:
            pred_labels[n] = 'A'

    predictions = list()

    for idx in range(len(X_test)):
        predictions.append((ecg_names[idx], pred_labels[idx]))
    # ------------------------------------------------------------------------------
    return predictions  # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
