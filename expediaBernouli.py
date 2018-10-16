# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:27:42 2017

@author: wenja
"""

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import pickle
import os


def map5eval(preds, dtrain):
    actual = dtrain.get_label()
    predicted = preds.argsort(axis=1)[:,-np.arange(5)]
    metric = 0.
    for i in range(5):
        metric += np.sum(actual==predicted[:,i])/(i+1)
    metric /= actual.shape[0]
    return 'MAP@5', -metric

train = pd.read_csv("train_n.csv")
test = pd.read_csv("test_n.csv")
#test = test.drop(id, axis=1)
destinations = pd.read_csv("feature_pca.csv")

#impute missing values
train.isnull().any()
train['orig_destination_distance'] = train.orig_destination_distance.fillna(train.orig_destination_distance.mean())
train['len_stay'] = train.len_stay.fillna(train.len_stay.mean())
test.isnull().any()
test['orig_destination_distance'] = test.orig_destination_distance.fillna(test.orig_destination_distance.mean())
test['len_stay'] = test.len_stay.fillna(test.len_stay.mean())

train = pd.merge(train, destinations, how='left', on='srch_destination_id')
test = pd.merge(test, destinations, how='left', on = 'srch_destination_id')
train['dest_1'] = train.dest_1.fillna(train.dest_1.mean())
train['dest_2'] = train.dest_1.fillna(train.dest_2.mean())
test['dest_1'] = test.dest_1.fillna(test.dest_1.mean())
test['dest_2'] = test.dest_1.fillna(test.dest_2.mean())
y = train.hotel_cluster
trainX = train.drop('hotel_cluster', axis=1)
test = test.drop('id', axis=1)

clf = BernoulliNB(alpha=1.0)
sw = 1+4*train.is_booking
clf.partial_fit(trainX, y, classes=np.arange(100), sample_weight=sw)

pickle.dump(clf, open('bernouliNB', 'wb'))
preds = clf.predict_proba(test)