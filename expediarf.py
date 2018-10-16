# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 15:54:23 2017

@author: wenja
"""

import numpy as np
import pandas as pd
import os

#mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.2.0-posix-seh-rt_v5-rev0\\mingw64\\bin'

#os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
#sns.set_style('whitegrid')
%matplotlib inline

# machine learning

from sklearn.ensemble import RandomForestClassifier


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
test = test.drop('id', axis=1)
destinations = pd.read_csv("feature_pca.csv")

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
ysmp = y.iloc[0:100000,]
trainXsmp = trainX.iloc[0:100000,]
#X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(trainX, y, stratify=y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
clf.fit(trainX, y)
#plot importance
feature = trainX.columns
importances = clf.feature_importances_
indices = np.argsort(importances)

plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), feature[indices])
plt.xlabel('Relative Importance')

pred_1 = clf.predict_proba(test.iloc[0:1000000,])
pred_2 = clf.predict_proba(test.iloc[1000000:2528243,])
pred = np.concatenate([pred_1,pred_2], axis=0)
col_ind = np.argsort(-pred, axis=1)[:,:5]
hc = [' '.join(row.astype(str)) for row in col_ind]
submission = pd.read_csv("sample_submission.csv")
sub=pd.DataFrame(data=hc, index=submission.id)
sub.reset_index(inplace=True)

sub.columns = submission.columns
sub.to_csv('predrf_sub.csv', index=False)

#test error 0.15177
