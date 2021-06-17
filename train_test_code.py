import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score,recall_score,precision_score
from wettbewerb import load_references



ecg_leads,ecg_labels,fs,ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz
# print('ecg_leads: ', ecg_leads)
# print('ecg_names: ', ecg_names)
# print('ecg_labels: ', ecg_labels)

from ecgdetectors import Detectors

detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors
sdnn_normal = np.array([])                                # Initialisierung der Feature-Arrays
sdnn_afib = np.array([])
sdnn_total =[]
ecg_leads_reduced = np.array([])
ecg_labels_reduced = np.array([])
for idx, ecg_lead in enumerate(ecg_leads):
    r_peaks = detectors.hamilton_detector(ecg_lead)     # Detektion der QRS-Komplexe
    sdnn = np.std(np.diff(r_peaks)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden
    sdnn_total = np.append(sdnn_total, sdnn)  # Kombination der beiden SDNN-Listen

    # if ecg_labels[idx]=='N':
    #   sdnn_normal = np.append(sdnn_normal, sdnn)         # Zuordnung zu "Normal"
    # if ecg_labels[idx]=='A':
    #   sdnn_afib = np.append(sdnn_afib, sdnn)             # Zuordnung zu "Vorhofflimmern"
    # if (ecg_labels[idx] == 'N') | (ecg_labels[idx] =='A'):
    #     ecg_labels_reduced = np.append(ecg_labels_reduced, ecg_labels[idx])
    #     ecg_leads_reduced = np.append(ecg_leads_reduced, sdnn)



print(sdnn_total.shape)

# # df means data frame
df = pd.DataFrame(sdnn_total).fillna(0) # To avoid NAN problems, fill 0 for the train matrices
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

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10,stratify=y)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=10)

import collections
result = collections.Counter(y_test)
print(result)

######## k nearest neighbor (KNN) ########
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2)
knn.fit(x_train, y_train)

knn_pred = knn.predict(x_test)
knn_cm = confusion_matrix(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred, pos_label='A')
knn_recall = recall_score(y_test, knn_pred, pos_label='A')
knn_f1 = f1_score(y_test, knn_pred, pos_label='A')
print("Accuracy:",accuracy_score(y_test, knn_pred))
print("confusion matrix", knn_cm)
print("precision score: ", knn_precision)
print("recall score: ", knn_recall)
print("F1 score: ", knn_f1)

####### Decision Tree ########
# from sklearn import tree
#
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(x_train, y_train)
# clf_pred = clf.predict(x_test)
# clf_cm = confusion_matrix(y_test, clf_pred)
# clf_precision = precision_score(y_test, clf_pred, pos_label='A')
# clf_recall = recall_score(y_test, clf_pred, pos_label='A')
# clf_f1 = f1_score(y_test, clf_pred, pos_label='A')
# print("Accuracy:",accuracy_score(y_test, clf_pred))
# print("confusion matrix", clf_cm)
# print("precision score: ", clf_precision)
# print("recall score: ", clf_recall)
# print("F1 score: ", clf_f1)


