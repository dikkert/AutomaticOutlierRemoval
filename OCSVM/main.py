from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import SGDOneClassSVM
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler 
import numpy as np
import laspy
import os
import pandas as pd

from  preprocessing import *
from  analysis import testAccuracy

PrepareOcsvmTestData("D:/OCSVM/test/","D:/OCSVM/test/")

preprocessing = extractandTrain()
preprocessing.extract_pc("D:/OCSVM/train")
preprocessing.ocsvmtrainer()
preprocessing.ocsvmpredict("D:/OCSVM/test")

testAccuracy("D:/OCSM/test", 12)

import laspy
import open3d as o3d

# Open the LAS file and read the point data
las_file = laspy.read("D:/OCSVM/test/BET_NB_GN_0-011.laz")
points = o3d.geometry.PointCloud()
points.points = o3d.utility.Vector3dVector(np.vstack((las_file.x, las_file.y, las_file.z)).transpose())

# Set the number of nearest neighbors to use
k = 10

# Compute the point density per point using KNN
distances, _ = points.compute_nearest_neighbor_distance()
volumes = (4/3) * np.pi * np.power(distances[:, k], 3)
point_density = (k + 1) / volumes

# Print the result
print("Point density per point:", point_density)