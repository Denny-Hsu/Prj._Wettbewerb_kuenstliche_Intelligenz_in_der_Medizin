# 18-ha-2010-pj
Code zu "Wettbewerb künstliche Intelligenz in der Medizin" SoSe 2021
<Projektseminar von Technische Universität Darmstadt>
<project seminar from Technical University of Darmstadt>

## Goal
To predict the **normal** and **atrial fibrillation** from ecg-signals.

## Setting before using.
1. Python's version is under 3.7.
2. If you still wanna use Python 3.8, must do some changes of the code in **train.py**:
    fit_resample -> fit_sample

## Steps
1. Use hrv_time to extract features from ecg-signals.
2. Since the quantity of label **N** and **A** is big difference, resample the training dataset.
3. Build the model from gradient boosting classifier.
4. Build the model from XXX.
5. Compare with these two models.
6. 
    
    
## Wichtig für die Abgabe!

Die Dateien 
- predict_pretrained.py
- predict_trained.py
- wettbewerb.py
- score.py

werden von uns beim testen auf den ursprünglichen Stand zurückgesetzt. Es ist deshalb nicht empfehlenswert diese zu verändern.

Bitte alle verwendeten packages in "requirements.txt" bei der Abgabe zur Evaluation angeben. Wir selbst verwenden Python 3.8. Wenn es ein Paket gibt, welches nur unter einer anderen Version funktioniert ist das auch in Ordung. In dem Fall bitte Python-Version mit angeben.
