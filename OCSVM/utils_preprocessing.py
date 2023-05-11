# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:07:53 2023

@author: Dick
"""

from jakteristics import FEATURE_NAMES
import np
import laspy
import os

class preprocessing_utils:
    
    def __init__(self):
        self.flist = ["x","y","z","red","green","blue","intensity"]+list(FEATURE_NAMES)
        
    def normalize_array(self,input):
        norm = np.linalg.norm(input)
        if norm == 0:
            return input
        return input / norm
    
    def feature_extraction(self,pathname,flist=self.flist):
        pc = laspy.read(pathname)
        self.nfeat = len(flist)
        npoints = len(pc.points)
        self.features = np.empty((npoints,0))
        for i in flist:
            i = np.array(pc[f"{i}"])
            preprocessing_utils.normalize_array(i)
            features = np.c_[features,i]
        print(features.shape)
        print(features[1])
        return self.features
    def input_path_iterator(self, directory, files=None, multimaps=False):
        if files is True:
            las_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith((".laz", ".las"))]
            file2_prefixes = []
            # Iterate through las files, run function body on 2 las files with 9 similar characters
            for i, file1_path in enumerate(las_files):
                if "outliers" not in file1_path:
                    prefix_length = len(os.path.basename(file1_path)[:-4])
                    file1_prefix = os.path.basename(file1_path)[:prefix_length]
                    for file2_path in las_files[i+1:]:
                        file2_prefix = os.path.basename(file2_path)[:prefix_length]
                        file2_prefixes.append(file2_prefix)
                        if file1_prefix in file2_prefix:
                            print(f"Running function on {file1_path} and {file2_path}")
                            filename = output_path + file1_prefix + "_outliers.npy"
                            if not os.path.exists(os.path.join(directory, filename)):
                                # Add the desired function body here
                                pass  # Placeholder for the function body
    
        if multimaps is True:
            for folder in os.listdir(directory):
                if os.path.isdir(os.path.join(directory, folder)):
                    folder_path = os.path.join(directory, folder)
                    for filename in os.listdir(folder_path):
                        if filename.endswith(".las") and "nocolor" not in filename:
                            path = os.path.join(folder_path, filename).replace("//", "/")
                            self.pathnames.append(path)
                else:
                    for filename in os.listdir(directory):
                        if filename.endswith(".las") and "nocolor" not in filename:
                            path = os.path.join(directory, filename).replace("//", "/")
                            if path not in self.pathnames:
                                self.pathnames.append(path)
    
                                