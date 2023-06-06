# -*- coding: utf-8 -*-
"""
Created on Sun May 21 22:16:30 2023

@author: Dick
"""

'''script to quantify outlier data '''

import laspy
import numpy as np
import matplotlib.pyplot as plt
import os
directory = "D:/outliers_classified"

def check_distribution(directory):
    pathnames = []
    count_jacob = 0
    count_mirror = 0
    count_object = 0
    count_noise = 0
    for folder in os.listdir(directory):
        if os.path.isdir(os.path.join(directory,folder)):
            folder = os.path.join(directory,folder)
            for filename in os.listdir(folder):
                if filename.endswith(".laz") and not "nocolor" in filename:
                    path = os.path.join(folder,filename).replace("//","/")
                    pathnames.append(path)
                    
        else:
            for filename in os.listdir(directory):
                if filename.endswith(".laz",) and not "nocolor" in filename:
                    path = os.path.join(directory,filename).replace("//","/")
                    if path not in pathnames:
                        pathnames.append(path)
                    
    
    # iterate over the point clouds and return an 
    
    for pathname in pathnames:
        try:
            file = laspy.read(pathname)
            print(f"reading{os.path.basename(pathname)}")
            if "classification" in file.point_format.dimension_names:
                # Retrieve the classification values as a numpy array
                classification = file.classification
                # jacob
                count_classification_1 = len(classification[classification == 1])
                count_jacob += count_classification_1
                # mirror
                count_classification_2 = len(classification[classification == 2])
                count_mirror += count_classification_2
                # count_obejct
                count_classification_3 = len(classification[classification == 3])
                count_object += count_classification_3
                # Count the number of points with classification 4 and 0
                count_classification_4 = len(classification[classification == 4])
                count_classification_0 = len(classification[classification == 0])
                count_noise += (count_classification_0 + count_classification_4)
        except:
            pass
    total_outlier = [count_jacob,count_mirror,count_object,count_noise]
    labels = ["person", "mirror", "object", "noise"]
    colors = ["black", "red", "grey", "mistyrose"]
    # Create the pie chart
    plt.pie(total_outlier, labels=labels, colors=colors, autopct='%1.1f%%')

    # Set the aspect ratio to be equal to make the pie circular
    plt.axis('equal')

    # Add a title
    plt.title("Oulier Classification Distribution")

    # Move percentage labels outside the pie chart
    plt.gca().set_position([0, 0, 0.5, 1])  # Adjust the position as needed

    # Display the chart
    plt.show()
    return total_outlier

check_distribution("D:/outliers_classified")

def check_feature_distribtutions(directory,feature):  #,outlier_types,feature):
    pathnames = []
    all_x = np.array([])
    intensity = 0
    colors = ['red', 'blue', 'green', 'purple'] 
    for folder in os.listdir(directory):
        if os.path.isdir(os.path.join(directory,folder)):
            folder = os.path.join(directory,folder)
            for filename in os.listdir(folder):
                if filename.endswith(".laz") and not "nocolor" in filename:
                    path = os.path.join(folder,filename).replace("//","/")
                    pathnames.append(path)
                    
        else:
            for filename in os.listdir(directory):
                if filename.endswith(".laz",) and not "nocolor" in filename:
                    path = os.path.join(directory,filename).replace("//","/")
                    if path not in pathnames:
                        pathnames.append(path)
                    
    
    # iterate over the point clouds and return an 
    #for idx, outlier_type in enumerate(outlier_types):
       # try:
    for pathname in pathnames:
        try:
            file = laspy.read(pathname)
            print(f"reading {os.path.basename(pathname)}")
            if "classification" in file.point_format.dimension_names:
                # Retrieve the classification values as a numpy array
                
                #classification = file.classification
            
                x_values =  np.asarray(file[feature])  #[classification == outlier_type])
                print(np.mean(x_values))
                x_values_normalized = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
                x_values_normalized = x_values_normalized[~np.isnan(x_values_normalized)]
                all_x = np.concatenate((all_x,x_values_normalized))
                print(len(all_x))
                
        except:
            pass
   
    hist, bin_edges = np.histogram(all_x, bins='auto')
    frequencies = hist / np.sum(hist)

    # Normalize the frequencies
    #normalized_frequencies = frequencies / np.sum(frequencies)
    # Set labels and title
    
   

    # Create a bar plot of the normalized frequencies
    plt.bar(bin_edges[:-1],frequencies, width=np.diff(bin_edges), align='edge')  #,label =outlier_type)
    plt.xlabel(f"{feature} Values")
    plt.ylabel("Frequency")
    plt.title(f"Normalized Frequency of {feature} Values")
    #plt.legend()
    # Show the plot
    plt.show()
features = ["X","Y","Z","red","green","blue","intensity"]
for i in features:    
    check_feature_distribtutions("D:/OCSVM/without_geometric_features/train/BET_OB_VL_N",i)
 total_outlier

file = laspy.read("D:/outliers_classified/BET_OB_GZ_1/BET_OB_GZ_1-034_outliers.laz")
# Filter points with classification 4
classification = file.classification
print(np.mean(x_values))
x_values = np.asarray(file.x[classification == 0]).T
x_values_normalized = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
print(len(x_values))
all_x = np.array([])
print(len(all_x))
all_x = np.concatenate((all_x,x_values))
print(len(x_values))
if np.isnan(x_values_normalized).any:
    print("ah)")
# Normalize the X values
x_values_normalized = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))

# Create a histogram of the normalized X values
plt.hist(x_values_normalized, bins=20)

# Set labels and title
plt.xlabel("Normalized X Values")
plt.ylabel("Frequency")
plt.title("Distribution of Normalized X Values (Classification 4)")

# Show the plot
plt.show()
hist, bin_edges = np.histogram(x_values, bins='auto')
frequencies = hist / np.sum(hist)

# Normalize the frequencies
normalized_frequencies = frequencies / np.sum(frequencies)
# Set labels and title
plt.xlabel("X Values")
plt.ylabel("Normalized Frequency")
plt.title("Normalized Frequency of X Values")

# Create a bar plot of the normalized frequencies
plt.bar(bin_edges[:-1],hist, width=np.diff(bin_edges), align='edge')


# Show the plot
plt.show()



