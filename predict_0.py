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

    ecg_leads,ecg_labels,fs,ecg_names = load_references('../test/') # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz

    def calculate_entropy(list_values):
        counter_values = Counter(list_values).most_common()
        probabilities = [elem[1] / len(list_values) for elem in counter_values]
        entropy = scipy.stats.entropy(probabilities)
        return entropy

    def calculate_statistics(list_values):
        n5 = np.nanpercentile(list_values, 5)
        n25 = np.nanpercentile(list_values, 25)
        n75 = np.nanpercentile(list_values, 75)
        n95 = np.nanpercentile(list_values, 95)
        median = np.nanpercentile(list_values, 50)
        mean = np.nanmean(list_values)
        std = np.nanstd(list_values)
        var = np.nanvar(list_values)
        rms = np.nanmean(np.sqrt(list_values ** 2))
        return [n5, n25, n75, n95, median, mean, std, var, rms]

    def calculate_crossings(list_values):
        zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
        no_zero_crossings = len(zero_crossing_indices)
        mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
        no_mean_crossings = len(mean_crossing_indices)
        return [no_zero_crossings, no_mean_crossings]

    def get_features(list_values):
        entropy = calculate_entropy(list_values)
        crossings = calculate_crossings(list_values)
        statistics = calculate_statistics(list_values)
        return [entropy] + crossings + statistics

    def get_ecg_features(ecg_data, ecg_labels, waveletname):
        list_features = []
        list_unique_labels = list(set(ecg_labels))
        list_labels = [list_unique_labels.index(elem) for elem in ecg_labels]
        for signal in ecg_data:
            list_coeff = pywt.wavedec(signal, waveletname)
            features = []
            for coeff in list_coeff:
                features += get_features(coeff)
            list_features.append(features)
        return list_features, list_labels

    df_test = pd.DataFrame(ecg_leads).fillna(0)  # To avoid NAN problems, fill 0 for the train matrices
    df_test['file_name'] = pd.Series(ecg_names)  # don't need .mat from file's name
    df_test['label'] = pd.Series(ecg_labels)
    dataset_test = df_test[(df_test["label"] == 'N') | (df_test["label"] == 'A')].reset_index(drop=True)  # only keep A and N
    print(dataset_test)
    name_test = dataset_test["file_name"].values
    lead_test = dataset_test.iloc[:, :-2].values  # all rows, columns without file_name and label
    label_test = dataset_test["label"].values


    x_test_ecg,y_test_ecg = get_ecg_features(lead_test, label_test, 'db4')

    pickle_in = open('cls.pickle', 'rb')
    cls_pred = pickle.load(pickle_in)

    # x_test_trans = cls_pred.transform(x_test_ecg)

    cls_over_pred = cls_pred.predict(x_test_ecg)


    def pred_use(test_name):

        pred = list()

        pred = np.append(pred, (list(test_name), list(cls_over_pred)))

        return pred

    predictions = pred_use(name_test)


    # ------------------------------------------------------------------------------
    return predictions  # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
