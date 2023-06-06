# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:24:55 2023

@author: Dick
"""

'''keep only 4'''

import laspy
import numpy as np
import os

def extract_jacob(directory, out_path):
    pathnames = []
    for folder in os.listdir(directory):
        if os.path.isdir(os.path.join(directory,folder)):
            folder = os.path.join(directory,folder)
            for filename in os.listdir(folder):
                if filename.endswith(".laz") and not "nocolor" in filename:
                    path = os.path.join(folder,filename).replace("//","/")
                    pathnames.append(path)
                    
        else:
            for filename in os.listdir(directory):
                if filename.endswith(".laz") and not "nocolor" in filename:
                    path = os.path.join(directory,filename).replace("//","/")
                    if path not in pathnames:
                        pathnames.append(path)
                        
        for path in pathnames:
            file =laspy.read(path)
            classification = file.classification
            file.points = file.points[classification == 1]
            new_file = laspy.create(point_format=file.header.point_format,file_version=file.header.version)
            new_file.points = file.points
            # save lasfile to specified directory using the input name
            filename = out_path+os.path.basename(path)[:-4]+"_jacob.laz"
            if not os.path.exists(os.path.join(out_path, filename)):
                new_file.write(f"{out_path}{os.path.basename(path)[:-4]}_jacob.laz")

extract_jacob("D:/outliers_classified/","D:/outliers_classified/jacob")

performance = {"OCSVM": 99.5,"CANUPO": x, "LOP": x}
training_size = {}
type_of_outlier_all_features = {"all": 0,995,"jacob": 0.9998776239570005,"mirror": 0.9998776239570005, "object": 0.9998776239570005, "noise": 1,0}
type_of_outlier_RGBI = {"all": 0,995,"jacob":0.9994309236818498,"mirror": 0.9994309236818498,"object": 0.9994309236818498,"noise": 0.9994310031728847}
import matplotlib.pyplot as plt
from sklearn import preprocessing
type_of_outlier_all_features = {
    "all": 0.995,
    "jacob": 0.9998776239570005,
    "mirror": 0.9998776239570005,
    "object": 0.9998776239570005,
    "noise": 1.0
}

type_of_outlier_RGBI = {
    "all": 0.995,
    "jacob": 0.9994309236818498,
    "mirror": 0.9994309236818498,
    "object": 0.9994309236818498,
    "noise": 0.9994310031728847
}
min_max_scaler = preprocessing.MinMaxScaler()
values_all_features = list(type_of_outlier_all_features.values())
values_all_features_normalized = min_max_scaler.fit_transform([[val] for val in values_all_features])
values_RGBI = list(type_of_outlier_RGBI.values())
values_RGBI_normalized = min_max_scaler.fit_transform([[val] for val in values_RGBI])

labels = list(type_of_outlier_all_features.keys())

x = range(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x, values_all_features_normalized.flatten(), width, label='All Features')
rects2 = ax.bar([i + width for i in x], values_RGBI_normalized.flatten(), width, label='RGBI')

ax.set_xlabel('Type of Outlier')
ax.set_ylabel('Normalized Value')
ax.set_title('Type of Outlier Comparison (Normalized)')
ax.set_xticks([i + width/2 for i in x])
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()

