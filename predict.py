# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os

from train import calculate_crossings, calculate_entropy, calculate_statistics, get_features, get_ecg_features
from train import cls

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


    df_test = pd.DataFrame(ecg_leads).fillna(0)  # To avoid NAN problems, fill 0 for the train matrices
    df_test['file_name'] = pd.Series(ecg_names)  # don't need .mat from file's name
    df_test['label'] = pd.Series(ecg_labels)
    dataset_test = df_test[(df_test["label"] == 'N') | (df_pred["label"] == 'A')].reset_index(drop=True)  # only keep A and N
    name_test = dataset_test["file_name"].values
    lead_test = dataset_test.iloc[:, :-2].values  # all rows, columns without file_name and label
    label_test = dataset_test["label"].values


    x_test_ecg,y_test_ecg = get_ecg_features(lead_test, label_test, 'db4')
    cls_over_pred = cls.predict(x_test_ecg)


    def pred_use(test_name):

        pred = list()

        pred = np.append(pred, (list(test_name), list(cls_over_pred)))

        return pred

    predictions = pred_use(name_test)


    # ------------------------------------------------------------------------------
    return predictions  # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!


        
