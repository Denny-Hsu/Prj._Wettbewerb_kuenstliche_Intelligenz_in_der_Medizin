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

    ecg_leads,ecg_labels,fs,ecg_names = load_references('../test/') # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name

    test_set = ecg_leads
    for i in tqdm(range(len(test_set))):
        data = ecg_leads[i].reshape(len(ecg_leads[i]), 1)
        plt.figure(figsize=(60, 5))
        plt.xlim(0, len(ecg_leads[i]))
        plt.plot(data, color='black', linewidth=0.1)
        plt.savefig('../ecg_images/{}.png'.format(ecg_names[i]))

    onlyfiles = [f for f in listdir("./ecg_images") if isfile(join("./ecg_images", f))]

    df = pd.read_csv("./test/REFERENCE.csv", header=None)
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

    test_images, test_labels = x, y
    # # Normalize pixel values to be between 0 and 1
    test_images = test_images / 255.0


    from keras.models import load_model

    model = load_model('best_model.h5', custom_objects={'fbeta': fbeta})
    pred_labels = model.predict_classes(test_images)

    for n, i in enumerate(pred_labels):
        if i == 0:
            y[n] = "N"
        elif i == 1:
            y[n] = "A"

    def pred_use(test_name):

        pred = list()

        pred = np.append(pred, (list(test_name), list(pred_labels)))

        return pred

    predictions = pred_use(name_test)



    # ------------------------------------------------------------------------------
    return predictions  # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
