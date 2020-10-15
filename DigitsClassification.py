
"""
Created on Thu Oct  8 11:25:29 2020

@author: Alfred Gaubrich

Modell zur Klassifizierung von Zahlen. Trainiert und validiert auf dem Sklearn Digits Datensatz. SVM als
"Black Box" - Klassifizierer
"""
# Benötigte Pakete
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Laden der Datensätze
digits = datasets.load_digits()

# Aufteilen in Test und Trainingsdaten
X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target)



# Klassifizierer
clf = svm.SVC(gamma = 0.001, C = 100)

# Fitten des Modells und prediction
clf.fit(X_train, Y_train)
Prediction = clf.predict(X_test)


# zählt die Anzahl der richtig klassifizierten Zahlen
counter = 0
for index in range(Prediction.size):
    if Prediction[index] == Y_test[index]:
        counter = counter +1
        
    
print("der Anteil an richtig klassifizierten Zahlen Beträgt " + str(counter / Prediction.size))
print(clf.score(X_test,Y_test))