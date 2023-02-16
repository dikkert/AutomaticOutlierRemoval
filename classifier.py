# import packages
import laspy
import numpy as np 
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

# load data using laspy
pc = laspy.read("D:/beneluxtunnel/diensttunnels/BET_NB_GZ_0/output/BET_NB_GZ_0-007.las")

# extract X,Y,Z coordinates
X = pc.X
Y = pc.Y
Z = pc.Z

# extract RGB values
R = pc.red
G = pc.green
B = pc.blue

# extract Intensity, 

features = np.vstack([X,Y,Z,R,G,B]).T
print(features)



