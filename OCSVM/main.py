from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import SGDOneClassSVM
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler 
import numpy as np
import laspy
import os
import pandas as pd

from preprocessing import normalize_array, PrepareOcsvmTestData, preprocessing
from analysis import testAccuracy

PrepareOcsvmTestData("D:/OCSVM/test","D:/OCSVM/test/")

preprocessing = preprocessing()
preprocessing.extract_pc("D:/OCSVM/train")
preprocessing.ocsvmtrainer()
preprocessing.ocsvmpredict("D:/OCSVM/test")

testAccuracy("D:/OCSM/test", 9)
