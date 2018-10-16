# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:48:13 2017

@author: wenja
"""

import numpy as np
import pandas as pd
import os

#mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.2.0-posix-seh-rt_v5-rev0\\mingw64\\bin'

#os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
sns.set_style('whitegrid')
%matplotlib inline


train = pd.read_csv("trainsmp.csv")
train = train.drop('Unnamed: 0', axis=1)
test = pd.read_csv("test.csv")

# the test only lacks two variable is_booking=1 and cnt also response variable hotel_cluster
#train2 = train[train.is_booking==1].drop(['is_booking','cnt'], axis=1)
#del train
#description analytics
# draw histogram for the hotel_cluster ranked by popularity
train["hotel_cluster"].value_counts().plot(kind='bar',colormap="Set3",figsize=(15,5))
fig, (axis1) = plt.subplots(1,1,figsize=(20,8))
sns.countplot('hotel_cluster',data=train,palette="Set3", ax=axis1)

#feature engineer for training and testing dataset
train['date_time'] = train.date_time.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
train['year'] = train.date_time.apply(lambda x:x.year)
train['month'] = train.date_time.apply(lambda x:x.month)
train['day'] = train.date_time.apply(lambda x:x.day)
train['hour'] = train.date_time.apply(lambda x:x.hour)
train['srch_co'] = pd.to_datetime(train['srch_co'], format="%Y-%m-%d")
train['srch_ci'] = pd.to_datetime(train['srch_ci'], format="%Y-%m-%d")
s = train['srch_co']-train['srch_ci']
train['len_stay'] = s.dt.days


test['date_time'] =  test.date_time.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
test['year'] = test.date_time.apply(lambda x:x.year)
test['month'] = test.date_time.apply(lambda x:x.month)
test['day'] = test.date_time.apply(lambda x:x.day)
test['hour'] = test.date_time.apply(lambda x:x.hour)
#choose median to replace the outlier
test.loc[test.srch_ci=="2161-10-00",'srch_ci'] = '2016-01-19'
test['srch_co'] = pd.to_datetime(test['srch_co'], format="%Y-%m-%d")
test['srch_ci'] = pd.to_datetime(test['srch_ci'], format="%Y-%m-%d")
s = test['srch_co']-test['srch_ci']
test['len_stay'] = s.dt.days
test['is_booking'] = 1
del s
train = train.drop(['date_time','srch_ci','srch_co','cnt'], axis=1)
test = test.drop(['date_time','srch_ci','srch_co'], axis=1)

pd.DataFrame(train).to_csv('train_n.csv',index=False)
pd.DataFrame(test).to_csv('test_n.csv', index=False)
#linkage solution, match location of th euser, origin-destination distance, htel market and search
#destination ID

feature2 = pd.read_csv("destinations.csv")
feature2 = feature2.drop('srch_destination_id', axis=1)
feature2_std = StandardScaler().fit_transform(feature2)
pca = PCA(n_components=2)
pca_fit = pca.fit(feature2_std)
print(pca_fit.explained_variance_ratio_)
# the first two components explanes most
feature2_pca = pd.DataFrame(pca_fit.fit_transform(feature2_std), columns=['dest_1', 'dest_2'])
feature2_pca['srch_destination_id']=feature2.srch_destination_id
pd.DataFrame(feature2_pca).to_csv('feature_pca.csv',index=False)


