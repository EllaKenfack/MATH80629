# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 16:01:02 2021

@author: Admin
"""

"""
Import necessary modules

"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sn
from inspect import getmembers

from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

"""
Import training and test dataset

"""

filename =r'C:\Users\Admin\Documents\Cours\MATH80629\Project\dataset.txt'
# filename_score='I:\Cours\dateststudent.txt'
dataset = pd.read_csv(filename,sep=",")
# datascore=pd.read_csv(filename_score,delimiter=" ")

print(dataset.head(5))

"""
Preparing Data For Training
"""

X = dataset.loc[:, ['V1','worry','chosen_emotion','text_long','text_short']]
y = dataset.iloc[:, 11].values

print(X.head(5))
X['chosen_emotion'].value_counts()

X['worry'].value_counts()

plt.hist(X['worry'], bins=9)
