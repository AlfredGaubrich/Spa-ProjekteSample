
"""
Created on Thu Oct  8 14:13:26 2020

@author: alfre
"""

import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn import datasets, svm
import matplotlib.pyplot as plt


X,y = datasets.load_digits(return_X_y = True)
svc = svm.SVC(kernel = 'linear')
C_s = np.logspace(-10,0,10)

scores_mean = list()
scores_standart_deviation = list()

for C in C_s:
    svc.C = C
    this_scores = cross_val_score(svc, X, y, cv =  KFold(n_splits=5))    
    scores_mean.append(np.mean(this_scores))
    scores_standart_deviation.append(np.std(this_scores))
    
    

# Plot
plt.figure()

plt.semilogx(C_s, scores_mean,)
plt.semilogx(C_s, np.array(scores_mean) + np.array(scores_standart_deviation), 'b--', )
plt.semilogx(C_s, np.array(scores_mean) - np.array(scores_standart_deviation), 'r--')
plt.xlabel('Parameter C')
plt.ylabel('CV Score')