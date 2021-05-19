import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from wettbewerb import load_references





ecg_leads,ecg_labels,fs,ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz
# print(ecg_labels)
print(ecg_names)

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
# temp =[]
# for file in load_mat_files[1:]:
#     print(file, 'is loading')
#     temp.append(sp.loadmat( data_path + file )['val'].tolist()[0] )
#     # print(dataset)
# # df means data frame
# df = pd.DataFrame(temp).fillna(0) # To avoid NAN problems, fill 0 for the train matrices
# df['file_name'] = pd.Series([file[:-4] for file in load_mat_files[1:]]) # don't need .mat from file's name
# ans = pd.read_csv('/Users/cheweihsu/Downloads/4.Semester-SS2021/Projektseminar-Wettbewerb Künsltiche Intelligenz in der Medizin/training/'.format(load_mat_files[0]), header=None)
# ans = ans[[0,1]].rename(columns={0:"file_name", 1:"label"})
# dataset = df.merge(ans, left_on="file_name", right_on="file_name")
# dataset = dataset[(dataset["label"] == 'N') | (dataset["label"] == 'A')].reset_index(drop=True) # only keep A and N
# # print(dataset)




