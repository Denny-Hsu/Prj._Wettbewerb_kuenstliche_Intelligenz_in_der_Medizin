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
import csv
import scipy.io as sp
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

'''
# Read the .mat file, test code
mat_contents = sp.loadmat("/Users/cheweihsu/Downloads/4.Semester-SS2021/Projektseminar-Wettbewerb Künsltiche Intelligenz in der Medizin/training/train_ecg_00001.mat")
print(mat_contents)
'''

# Give data path and save the matrices in temp, which means temporary
data_path = ('/Users/cheweihsu/Downloads/4.Semester-SS2021/Projektseminar-Wettbewerb Künsltiche Intelligenz in der Medizin/training/')
load_mat_files = os.listdir(data_path)
temp =[]

for file in load_mat_files[1:]:
    print(file, 'is loading')
    temp.append(sp.loadmat( data_path + file )['val'].tolist()[0] )
    # print(dataset)

# df means data frame
df = pd.DataFrame(temp).fillna(0) # To avoid NAN problems, fill 0 for the train matrices
df['file_name'] = pd.Series([file[:-4] for file in load_mat_files[1:]]) # don't need .mat from file's name
ans = pd.read_csv('/Users/cheweihsu/Downloads/4.Semester-SS2021/Projektseminar-Wettbewerb Künsltiche Intelligenz in der Medizin/training/'.format(load_mat_files[0]), header=None)
ans = ans[[0,1]].rename(columns={0:"file_name", 1:"label"})
dataset = df.merge(ans, left_on="file_name", right_on="file_name")
dataset = dataset[(dataset["label"] == 'N') | (dataset["label"] == 'A')].reset_index(drop=True) # only keep A and N
# print(dataset)


# Divide dataset into train-,valid- and test-dataset
# train,valid,test = 60%,20%,20%
X = dataset.iloc[:, :-2].values # all rows, columns without file_name and label
y = dataset["labels"].values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10,stratify=y)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=10)


######## k nearest neighbor (KNN) ########
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
mse_knn = mean_squared_error(y_test, knn_pred, squared=True)
print("Mean Squared Error: ", mse_knn)

######## Linear Regression (LR) ########
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

lr = LogisticRegression(random_state=0).fit(x_train, y_train)
lr_pred = lr.predict(x_test)
mse_lr = mean_squared_error(y_test, lr_pred, squared=True)
print("Mean Squared Error: ", mse_lr)

######## Decision Tree Regression ########
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# Create a decision tree regressor object from DecisionTreeRegressor class
DtReg = DecisionTreeRegressor(random_state= 0)
# Fit the decision tree regressor with training represented by x_train, y_train
DtReg.fit(x_train, y_train)
# Predicted from test dataset wrt Decision Tree Regression
DtReg_predict = DtReg.predict((x_test))
#Model Evaluation using R-Square for Decision Tree Regression
r_square = metrics.r2_score(y_test, DtReg_predict)
print('R-Square Error associated with Decision Tree Regression is:', r_square)
mse_DtReg = mean_squared_error(y_test, DtReg_pred, squared=True)
print("Mean Squared Error: ", mse_DtReg)


####### Support Vector Machine (SVM) ########
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

svc = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svc.fit(x_train, y_train)
svc_pred = svc.predict(x_test)
mse_svc = mean_squared_error(y_test, svc_pred, squared=True)
print("Mean Squared Error: ", mse_svc)

####### Tree ########
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
clf_pred = clf.predict(x_test)
mse_clf = mean_squared_error(y_test, clf_pred, squared=True)
print("Mean Squared Error: ", mse_clf)