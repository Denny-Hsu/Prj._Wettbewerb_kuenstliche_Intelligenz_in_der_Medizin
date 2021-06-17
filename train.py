# # -*- coding: utf-8 -*-
# """
# Beispiel Code und  Spielwiese
#
# """
# # Beispiel verwendet "SDNN" und "F1", so F1 weiter verwendet wird
#
# import csv
# import scipy.io as sio
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from wettbewerb import load_references
#
#  if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
#
# ecg_leads,ecg_labels,fs,ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz
#
# detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors
# sdnn_normal = np.array([])                                # Initialisierung der Feature-Arrays
# sdnn_afib = np.array([])
# for idx, ecg_lead in enumerate(ecg_leads):
#     r_peaks = detectors.hamilton_detector(ecg_lead)     # Detektion der QRS-Komplexe
#     sdnn = np.std(np.diff(r_peaks)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden
#     if ecg_labels[idx]=='N':
#       sdnn_normal = np.append(sdnn_normal,sdnn)         # Zuordnung zu "Normal"
#     if ecg_labels[idx]=='A':
#       sdnn_afib = np.append(sdnn_afib,sdnn)             # Zuordnung zu "Vorhofflimmern"
#     if (idx % 100)==0:
#       print(str(idx) + "\t EKG Signale wurden verarbeitet.")
#
# fig, axs = plt.subplots(2,1, constrained_layout=True)
# axs[0].hist(sdnn_normal,2000)
# axs[0].set_xlim([0, 300])
# axs[0].set_title("Normal")
# axs[0].set_xlabel("SDNN (ms)")
# axs[0].set_ylabel("Anzahl")
# axs[1].hist(sdnn_afib,300)
# axs[1].set_xlim([0, 300])
# axs[1].set_title("Vorhofflimmern")
# axs[1].set_xlabel("SDNN (ms)")
# axs[1].set_ylabel("Anzahl")
# plt.show()
#
# sdnn_total = np.append(sdnn_normal,sdnn_afib) # Kombination der beiden SDNN-Listen
# p05 = np.nanpercentile(sdnn_total,5)          # untere Schwelle
# p95 = np.nanpercentile(sdnn_total,95)         # obere Schwelle
# thresholds = np.linspace(p05, p95, num=20)    # Liste aller möglichen Schwellwerte
# F1 = np.array([])
# for th in thresholds:
#   TP = np.sum(sdnn_afib>=th)                  # Richtig Positiv
#   TN = np.sum(sdnn_normal<th)                 # Richtig Negativ
#   FP = np.sum(sdnn_normal>=th)                # Falsch Positiv
#   FN = np.sum(sdnn_afib<th)                   # Falsch Negativ
#   F1 = np.append(F1, TP / (TP + 1/2*(FP+FN))) # Berechnung des F1-Scores
#
# th_opt=thresholds[np.argmax(F1)]              # Bestimmung des Schwellwertes mit dem höchsten F1-Score
#
# if os.path.exists("model.npy"):
#     os.remove("model.npy")
# with open('model.npy', 'wb') as f:
#     np.save(f, th_opt)
#
# fig, ax = plt.subplots()
# ax.plot(thresholds,F1)
# ax.plot(th_opt,F1[np.argmax(F1)],'xr')
# ax.set_title("Schwellwert")
# ax.set_xlabel("SDNN (ms)")
# ax.set_ylabel("F1")
# plt.show()
#
# fig, axs = plt.subplots(2,1, constrained_layout=True)
# axs[0].hist(sdnn_normal,2000)
# axs[0].set_xlim([0, 300])
# tmp = axs[0].get_ylim()
# axs[0].plot([th_opt,th_opt],[0,10000])
# axs[0].set_ylim(tmp)
# axs[0].set_title("Normal")
# axs[0].set_xlabel("SDNN (ms)")
# axs[0].set_ylabel("Anzahl")
# axs[1].hist(sdnn_afib,300)
# axs[1].set_xlim([0, 300])
# tmp = axs[1].get_ylim()
# axs[1].plot([th_opt,th_opt],[0,10000])
# axs[1].set_ylim(tmp)
# axs[1].set_title("Vorhofflimmern")
# axs[1].set_xlabel("SDNN (ms)")
# axs[1].set_ylabel("Anzahl")
# plt.show()


import numpy as np
# import csv
# import scipy.io as sp
# import os
# import matplotlib.pyplot as plt
import pywt
import scipy.stats
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from collections import Counter #, defaultdict

from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

import collections

from wettbewerb import load_references


import scipy.signal as sps
# Read the .mat file, test code
# mat_contents = sp.loadmat("../training/train_ecg_00001.mat")
# print(mat_contents['val'])
# print(mat_contents['val'].shape)
# d_ecg = np.diff(mat_contents['val'].flatten())
# peaks_d_ecg = sps.find_peaks(d_ecg)
# print(peaks_d_ecg)
# print(np.array(peaks_d_ecg).shape)

# Give data path and save the matrices in temp, which means temporary
# data_path = ('../training/')
# load_mat_files = os.listdir(data_path)
# temp =[]

# for file in load_mat_files[1:]:
#     print(file, 'is loading')
#     temp.append(sp.loadmat( data_path + file )['val'].tolist()[0] )
    # print(dataset)




# df means data frame
# df = pd.DataFrame(temp).fillna(0) # To avoid NAN problems, fill 0 for the train matrices
# df['file_name'] = pd.Series([file[:-4] for file in load_mat_files[1:]]) # don't need .mat from file's name
# ans = pd.read_csv('../training/'.format(load_mat_files[0]), header=None)
# ans = ans[[0,1]].rename(columns={0:"file_name", 1:"label"})
# dataset = df.merge(ans, left_on="file_name", right_on="file_name")
# dataset = dataset[(dataset["label"] == 'N') | (dataset["label"] == 'A')].reset_index(drop=True) # only keep A and N
# print(dataset)


# Divide dataset into train-,valid- and test-dataset
# train,valid,test = 60%,20%,20%
# X = dataset.iloc[:, :-2].values # all rows, columns without file_name and label
# y = dataset["labels"].values
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10,stratify=y)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=10)


######## k nearest neighbor (KNN) ########
# from sklearn.neighbors import KNeighborsClassifier
#
# knn = KNeighborsClassifier(n_neighbors = 9, metric = 'minkowski', p = 2)
# knn.fit(x_train, y_train)
# knn_pred = knn.predict(x_test)
# knn_cm = confusion_matrix(y_test, knn_pred)
# print("Accuracy:",accuracy_score(y_test, knn_pred))
# print("confusion matrix", knn_cm)


## Find the best n_neighbors
# for i in range(2,11):
#     knn = KNeighborsClassifier(n_neighbors = i, metric = 'minkowski', p = 2)
#     knn.fit(x_train, y_train)
#     knn_pred = knn.predict(x_test)
#     knn_cm = confusion_matrix(y_test, knn_pred)
#     knn_precision = precision_score(y_test, knn_pred)
#     knn_recall = recall_score(y_test, knn_pred)
#     knn_f1 = f1_score(y_test, knn_pred)
#     print("==========={}==============".format(i))
#     print("Accuracy:", accuracy_score(y_test, knn_pred))
#     print("confusion matrix: ", knn_cm)
#     print("precision score: ",knn_precision)
#     print("recall score: ",knn_recall)
#     print("F1 score: ",knn_f1)
  
# -> n = 9 has the best recall and accuracy

# Calculate the numbers of N and A, in order to clarify which one belongs to TP or TN
# result = dict((i, y_test.count(i)) for i in y_test)
# print(result)
# -> Result is like score.py, i.e. TP is A and TN is N

####### Decision Tree ########
# from sklearn import tree
#
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(x_train, y_train)
# clf_pred = clf.predict(x_test)
# clf_cm = confusion_matrix(y_test, clf_pred)
# print("Accuracy:",accuracy_score(y_test, clf_pred))
# print("confusion matrix", clf_cm)


ecg_leads,ecg_labels,fs,ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz
# print('ecg_leads: ', ecg_leads)
# print('ecg_names: ', ecg_names)
# print('ecg_labels: ', ecg_labels)


# detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors
# sdnn_normal = np.array([])                                # Initialisierung der Feature-Arrays
# sdnn_afib = np.array([])
# sdnn_total =[]
#
# for idx, ecg_lead in enumerate(ecg_leads):
#     r_peaks = detectors.hamilton_detector(ecg_lead)     # Detektion der QRS-Komplexe
#     sdnn = np.std(np.diff(r_peaks)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden
#
#     if ecg_labels[idx]=='N':
#         sdnn_normal = np.append(sdnn_normal, sdnn)         # Zuordnung zu "Normal"
#     if ecg_labels[idx]=='A':
#         sdnn_afib = np.append(sdnn_afib, sdnn)             # Zuordnung zu "Vorhofflimmern"



# # df means data frame
df = pd.DataFrame(ecg_leads).fillna(0) # To avoid NAN problems, fill 0 for the train matrices
df['file_name'] = pd.Series(ecg_names) # don't need .mat from file's name
# print('df: ',df)
df['label'] = pd.Series(ecg_labels)
# print('df: ',df)
dataset = df[(df["label"] == 'N') | (df["label"] == 'A')].reset_index(drop=True) # only keep A and N
print(dataset)


# Divide dataset into train-,valid- and test-dataset
# train,valid,test = 60%,20%,20%
X = dataset.iloc[:, :-2].values # all rows, columns without file_name and label
y = dataset["label"].values
# print('X:',X.shape,'\n',X)
# print('y:',y.shape)

# X = X.reshape(X.shape[0], X.shape[1], 1)

for n, i in enumerate(y):
    if i == "N":
        y[n] = 0
    elif i == "A":
        y[n] = 1

# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10,stratify=y)
# x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=10)

result = collections.Counter(y)
print(result)

# import os
# import time
# import pandas as pd
# import scipy.io as sio
# from scipy.fftpack import fft
# from IPython.display import display



# import datetime as dt



###########################################################
def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
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
    rms = np.nanmean(np.sqrt(list_values**2))
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

def get_uci_har_features(dataset, labels, waveletname):
    uci_har_features = []
    for signal_no in range(0, len(dataset)):
        features = []
        for signal_comp in range(0,dataset.shape[2]):
            signal = dataset[signal_no, :, signal_comp]
            list_coeff = pywt.wavedec(signal, waveletname)
            for coeff in list_coeff:
                features += get_features(coeff)
        uci_har_features.append(features)
    X = np.array(uci_har_features)
    Y = np.array(labels)
    return X, Y

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

x_over, y_over = SMOTE(random_state=42).fit_resample(list(X), list(y))
X_over_ecg, Y_over_ecg = get_ecg_features(x_over, y_over, 'db4')

### GradientBoostingClassifier
cls = GradientBoostingClassifier(n_estimators=10000)
cls.fit(list(X_over_ecg), list(Y_over_ecg))



# f1_cls = f1_score(list(Y_test_ecg), list(cls_over_pred))
# print('The f1_cls:', f1_cls)
# cls_cm=confusion_matrix(list(Y_test_ecg), list(cls_over_pred))
# print('cls_cm: ',cls_cm)




