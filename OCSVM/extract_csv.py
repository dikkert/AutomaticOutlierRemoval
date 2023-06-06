# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:47:15 2023

@author: Dick
"""

import csv
from scipy import stats
import numpy as np
import laspy

acc_xyzrgbi_5 = []
acc_xyzrgbi_10 = []
acc_xyzrgbi_15 = []
acc_xyz_5 = []
acc_xyz_10 = []
acc_xyz_15 = []
acc_rgbi_5 = []
acc_rgbi_10 = []
acc_rgbi_15 = []
acc_linh = []

def extract_csv(input_file,acc_list):
    with open(input_file,"r") as file:
        reader = csv.reader(file)
        next(reader,None)
        for row in reader:
            acc_list.append(row[1])

extract_csv("D:/AutoMaticOutlierRemoval/OCSVM/evaluation_results_linh.csv", acc_linh)
print(acc_linh)
5

np_acc_rgbi_15 = np.asarray(acc_rgbi_15)
np_acc_rgbi_15 = np_acc_rgbi_15.astype(float)
np_acc_rgbi_5 = np.asarray(acc_rgbi_5)
np_acc_rgbi_5 = np_acc_rgbi_5.astype(float)
t_statistic,p_value = stats.ttest_ind(np_acc_rgbi_15,np_acc_rgbi_5)
print(p_value)
alpha = 0.05
if p_value < alpha:
    print("The two datasets are significantly different")
    

s_rgbi_5 = []
s_rgbi_10 = []
s_rgbi_15 = []
s_linh = []

def extract_csv_sensitivity(input_file,acc_list):
    with open(input_file,"r") as file:
        reader = csv.reader(file)
        next(reader,None)
        for row in reader:
            try: 
                bla = float(row[3])+float(row[4])+float(row[5])
                sensitivity = float(row[3])/ float(bla)
                acc_list.append(sensitivity)
            except:
                acc_list.append(0)

extract_csv_sensitivity("D:/AutoMaticOutlierRemoval/OCSVM/evaluation_results_linh.csv",s_linh)
print(s_rgbi_15)
print(mean(s_rgbi_15))
s_rgbi_15_no0 = [x for x in s_rgbi_15 if x != 0.0]
print(s_linh_no0)
np_s_rgbi_15 = np.asarray(s_rgbi_15)
np_s_rgbi_15 = np_s_rgbi_15.astype(float)
np_s_rgbi_5 = np.asarray(s_rgbi_5)
s_rgbi_5 = np_acc_rgbi_5.astype(float)
t_statistic,p_value = stats.ttest_ind(np_acc_rgbi_15,np_acc_rgbi_5)
print(p_value)
alpha = 0.05
if p_value < alpha:
    print("The two datasets are significantly different")
    
linh1 = []
linh4no0 =  [x for x in linh4 if x != 0.0]
linh2 = []
linh3 = []
linh4 = []
extract_csv_sensitivity("D:/AutoMaticOutlierRemoval/OCSVM/evaluation_results_linh_4.csv",linh4)
print(mean(linh4no0))

ocsvm1 = []
ocsvm4no0 = [x for x in ocsvm4 if x != 0.0]
ocsvm2 = []
ocsvm3 = []
ocsvm4 = []
extract_csv("D:/AutoMaticOutlierRemoval/OCSVM/evaluation_results_type4.csv",ocsvm4)
print(mean(ocsvm4no0))
np_ocsvm1 = np.asarray(ocsvm1)
np_ocsvm1 =np_ocsvm1.astype(float)
np_ocsvm3 = np.asarray(ocsvm3)
np_ocsvm3 = np_ocsvm3.astype(float)
t_statistic,p_value = stats.ttest_ind(np_ocsvm2,np_ocsvm4)
print(p_value)


pc = laspy.read("D:\OCSVM\Canupo\BET_NB_GN_0-030.laz")
classification = np.asarray(pc["CANUPO.class"])
 result = np.where(classification == 0.0,-1,1)
 np.save("D:/OCSVM/Canupo/BET_NB_0-030_canupo.npy",result)

import matplotlib.pyplot as plt

methods = ["LOP", "OCSVM", "Canupo"]
scores = [15, 0.3, 1.2]

plt.bar(methods, scores)
plt.title("Performance of outlier removal methods in percentage of outliers removed")
plt.xlabel("Methods")
plt.ylabel("Percentage of outliers removed")
plt.show()


outlier_type = ["person", "mirroring","object", "noise"]
scores = [0.6,0.9,0.7,0.01]
plt.bar(outlier_type, scores)
plt.title("Comparison of LOP performance on different types of outliers")
plt.xlabel("Type of outlier")
plt.ylabel("Percentage of outliers removed")
plt.show()

dataxyzrgbi = [
    0.9979766419634856,
 0.9979766419634856,
    0.9979766419634856
]

dataxyz = [
    0.9979766419634856,
    0.9979766419634856,
    0.9979766419634856
]

datargbi = [0.9961284055655696,
  0.9965990242221676,
    0.995030486299597]



datasize = [5,10,15]

fig, ax = plt.subplots()

# Plot the existing bars
ax.plot(datasize,dataxyzrgbi, label = "xyzrgbi")
ax.plot(datasize,dataxyz,label="xyz")
ax.plot(datasize,dataxyz,label = "rgbi")
# Set labels and title
ax.set_xlabel('Dataset size in folders')
ax.set_ylabel('Accuracy of inliers')
ax.set_title('Size of Dataset plotted against selected features')

# Show the plot
plt.show()

fig, ax = plt.subplots()

# Calculate the width of each bar
bar_width = 0.2

# Plot the bars for xyzrgbi
ax.bar([size - bar_width for size in datasize], dataxyzrgbi, width=bar_width, label="xyzrgbi")

# Plot the bars for xyz
ax.bar(datasize, dataxyz, width=bar_width, label="xyz")

# Plot the bars for rgbi
ax.bar([size + bar_width for size in datasize], datargbi, width=bar_width, label="rgbi")

# Set labels and title
ax.set_xlabel('Dataset size in folders')
ax.set_ylabel('Accuracy of inliers')
ax.set_title('Size of Dataset plotted against selected features')

# Set the x-axis ticks
ax.set_xticks(datasize)

# Set the legend
ax.legend()

# Show the plot
plt.show()