# 18-ha-2010-pj
Code zu "Wettbewerb künstliche Intelligenz in der Medizin" SoSe 2021

<Projektseminar von Technische Universität Darmstadt>

## Goal
Predict the **normal** and **atrial fibrillation** from ecg-signals, and try to get a better F1-Score.

## Setting
1. Python's version is under 3.7.
2. If use Python 3.8, must do some changes of the code in **train.py**: fit_resample -> fit_sample

## Steps
1. Use hrv_time to extract features from ecg-signals.
2. Since the quantity of label **N** and **A** is imbalanced in training dataset, resample it.
3. Build the model from Gradient Boosting Classifier.
4. Build the model from AdaBoost Classifier.
5. Build the model from XGBoost Classifier.
6. Build the model from CNN.
7. Compare with these four models.
8. Last but not least, choose the one, which has a better F1-Score.

## By Testing
1. Run train.py (Here should choose the model)
2. Run predict.py (Here should choose the model)
3. Run predict_pretrained.py (Call predict_label() and print .csv file for the prediction)
4. Run wettbewerb.py
5. Run score.py
  
