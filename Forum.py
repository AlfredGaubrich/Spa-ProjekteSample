# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 15:50:33 2020

@author: alfre
"""


from sklearn.datasets import fetch_20newsgroups


categories = [ 'alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(categories = categories)

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

from sklearn.feature_extraction.text import TfidfTransformer

tf_transformer = TfidfTransformer()

Transformed_X_train_counts = tf_transformer.fit_transform(X_train_counts)

x = 2


            