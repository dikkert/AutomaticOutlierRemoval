from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os
from statistics import mean
import matplotlib.pyplot as plt

def testAccuracy(input_path,input_path2):
    error_files = []
    total_acc = []
    total_TN = []
    total_cm = np.empty((2,2))
    i = 0
    variation = []
     # create lists of input paths
    npy_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".npy")]
    npy_files2 = [os.path.join(input_path2, f) for f in os.listdir(input_path2) if f.endswith(".npy")]
    # check for similar files
    for i, file1_path in enumerate(npy_files):
        file1_path = str(file1_path.replace("\\","/"))
        prefix_length = len(os.path.basename(file1_path)[:-4])
        print(file1_path)
        file1_prefix = os.path.basename(file1_path)[:15]
        for file2_path in npy_files2:
            file2_path = str(file2_path.replace("\\","/"))
            file2_prefix = os.path.basename(file2_path)
            if file1_prefix in file2_prefix:
                ### body of function
                print(f"Running function on {file1_path} and {file2_path}")
                # open training file and check for variation
                train = np.load(file1_path).astype(int)
                print("loading training file")
                if not np.all(train == 1):
                    variation.append(file1_path)
                # open test file    
                test = np.load(file2_path).astype(int)
                test = test
                print("loading testing file")
                # control structure to check for files with a different length 
                if train.shape[0] != test.shape[0]:
                    error_files.append(file1_path)
                ### analysis part of function
                else: 
                    # get accuracy score
                    acc = accuracy_score(train,test)
                    print("accuracy score is "+str(acc))
                    total_acc.append(acc)
                    # make a confusion matrix
                    cm = confusion_matrix(train,test, normalize="all")
                    np.add(total_cm,cm)
                    print(cm)
    # make a plot of the total confusion matrix                 
    disp = ConfusionMatrixDisplay(total_cm)
    disp.plot()
    plt.show()
    return mean(total_acc), error_files, variation
        

testAccuracy("D:/OCSVM/test/numpy/all_geom_features","D:/OCSVM/validation(in+out)/numpy")

import pandas as pd
model = np.load("D:/OCSVM/test/numpy/BET_NB_GN_0-049.npy")
np.all(model == 1)

test = np.load("D:/OCSVM/validation(in+out)/numpy/BET_NB_GN_0-030_outliers.npy")
npy_files2 = [os.path.join("D:/OCSVM/validation(in+out)/numpy", f) for f in os.listdir("D:/OCSVM/validation(in+out)/numpy") if f.endswith(".npy")]
for file2_path in npy_files2:
    file2_path = str(file2_path.replace("\\","/"))
    print(file2_path)
    test = np.load(file2_path)
    print(test[:10])
print(npy_files2)
accuracy_score(model,test)
print(model[:,-1])
print(test[:100])  
modeldf = pd.DataFrame(model)
print(modeldf)
accuracy_score(model[:,-1],test)

f1_score(model[:,-1], test)

import laspy
inlier = laspy.read("D:/OCSVM/validation(in+out)/BET_NB_GN_0-033_outliers.laz")
outlier = laspy.read("D:/OCSVM/validation(in+out)/BET_NB_GN_0-033_outliers.laz")
full = laspy.read("D:/OCSVM/test/BET_NB_GN_0-033.laz")
inpoint+outpoints == fullpoints
inpoint = len(inlier.points)
outpoints = len(outlier.points)
fullpoints = len(full.points)

 