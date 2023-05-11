# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:26:58 2023

@author: Dick
"""
from sklearn.linear_model import SGDOneClassSVM 
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from utils_preprocessing import preprocessing_utils
import numpy as np

class ocsvm:
    def extract_train(self,directory,batch_size=10000):
        self.ocsvm= SGDOneClassSVM()
        # creates a list of paths to files to read the point clouds
        self.pathnames = []
        preprocessing_utils.input_path_iterator(directory)
        for pathname in self.pathnames:
            print("extracting features of"+ pathname)
            preprocessing_utils.feature_extraction(self, pathname)
            scalar = StandardScaler()
            scalar.fit_transform(preprocessing_utils.features)
            print("fitting on "+pathname)
            self.ocsvm.partial_fit(preprocessing.features)
        
oc = ocsvm()
oc.extract_train("D:/OCSVM/with_geometric_features/train",)