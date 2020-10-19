# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 15:50:33 2020

@author: alfre
"""


from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB

categories = [ 'alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(categories = categories)

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

from sklearn.feature_extraction.text import TfidfTransformer

tf_transformer = TfidfTransformer()

Transformed_X_train_counts = tf_transformer.fit_transform(X_train_counts)


# Klassifizierer

clf = MultinomialNB().fit(Transformed_X_train_counts, twenty_train.target)

# Predicted new documents

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_transformed = tf_transformer.fit_transform(X_new_counts)
            
predicted = clf.predict(X_new_transformed)

# Ausgabe prediction

for doc, category in zip(docs_new, predicted):
    print(doc)
    print(twenty_train.target_names[category])