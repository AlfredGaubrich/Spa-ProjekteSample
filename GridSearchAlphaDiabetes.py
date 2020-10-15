# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 12:57:04 2020

@author: alfre
"""

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split#
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()


lasso = Lasso( random_state = 0, max_iter = 10000)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{'alpha' : alphas}]
n_folds = 5

clf = GridSearchCV(lasso, param_grid = tuned_parameters, cv = n_folds)
clf.fit(diabetes.data, diabetes.target)

scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']

plt.figure()
plt.semilogx(alphas,scores)
plt.semilogx(alphas, scores + scores_std, 'b--')
plt.semilogx(alphas, scores - scores_std, 'b--')

plt.axhline(np.max(scores), linestyle = '--', color = 'r')
plt.xlabel('alphas')
plt.ylabel('Mean test score +/- Std test score')