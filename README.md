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
5. Build the model from CNN.
6. Compare with these two models.
7. Last but not least, choose the one, which has a better F1-Score.
    
    
## Wichtig für die Abgabe!

Die Dateien 
- predict_pretrained.py
- predict_trained.py
- wettbewerb.py
- score.py

werden von uns beim testen auf den ursprünglichen Stand zurückgesetzt. Es ist deshalb nicht empfehlenswert diese zu verändern.

Bitte alle verwendeten packages in "requirements.txt" bei der Abgabe zur Evaluation angeben. Wir selbst verwenden Python 3.8. Wenn es ein Paket gibt, welches nur unter einer anderen Version funktioniert ist das auch in Ordung. In dem Fall bitte Python-Version mit angeben.
