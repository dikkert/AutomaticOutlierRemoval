# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:11:05 2023

@author: Dick
"""

'''preprocessing for specific class'''

from sklearn.linear_model import SGDOneClassSVM 
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import laspy
import joblib
from joblib import dump
from  jakteristics import FEATURE_NAMES

# Loop over each LAS file and concatenate its points with the points for the corresponding prefix
def PrepareOcsvmTestData(directory, output_path,outlier_type):
    '''function that reads all .las files in a directory and returns a classified by source ID .las file containing '''
    # list all .las files
    las_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith((".laz",".las"))]
    file2_prefixes = []
    # iterate through las files, run function body on 2 las files with 9 similar characters
    for i, file1_path in enumerate(las_files):
        if "outliers" not in file1_path: 
            prefix_length = len(os.path.basename(file1_path)[:-4])
            file1_prefix = os.path.basename(file1_path)[:prefix_length]
            for file2_path in las_files[i+1:]:
                file2_prefix = os.path.basename(file2_path)[:prefix_length]
                file2_prefixes.append(file2_prefixes)
                if file1_prefix in file2_prefix:
                    print(f"Running function on {file1_path} and {file2_path}")
                    filename = output_path+file1_prefix+"_outliers.npy"
                    if not os.path.exists(os.path.join(directory, filename)):
        # function body, read las files, concatenate points, write new .las file containing combined points, classified with source IDs
                    # read and extract all file features for file 1 and file 2
                        file1 = laspy.read(file1_path)
                        # X = np.array(file1.x)
                        # any_nan = np.isnan(X).any()
                        # if any_nan is True:
                        #     print("some of x is Nan")
                        #     break
                        # Intensity1 = np.array(file1.intensity)
    
                        # features1 = np.c_[R,G,B,Intensity]
                        file2 = laspy.read(file2_path)
                        classification = file2.classification

                        # Intensity2 = np.array(file2.intensity)

                        # combined_points = np.vstack((intensity1, intensity2))
                        # print("combine points")
                        result = np.where(classification == outlier_type, 1, -1)
                        combined_source_ids = np.concatenate((np.ones((len(file1.points),),dtype=int),result))
                        #combined_full = np.c_[combined_points,combined_source_ids]
                        # write to binary
                        outfile = "{}{}_outliers_{}".format(output_path,file1_prefix,outlier_type)
                        np.save(outfile,combined_source_ids)

    for i, file1_path in enumerate(las_files):
        if "outliers" not in file1_path: 
            prefix_length = len(os.path.basename(file1_path)[:-4])
            file1_prefix = os.path.basename(file1_path)[:prefix_length]
            if file1_prefix not in file2_prefixes:
                filename = output_path+file1_prefix+"_outliers.npy"
                if not os.path.exists(os.path.join(directory, filename)):
                    print("running function on "+file1_prefix)
                    file1 = laspy.read(file1_path)
                    combined_source_ids = np.ones((len(file1.points),),dtype=int)
                    # write to binary
                    np.save(f"{output_path}{file1_prefix}_outliers(nochange)",combined_source_ids)
outlier_types = [0,1,2,3]
for i in outlier_types:           
    PrepareOcsvmTestData("D:/OCSVM/without_geometric_features/validation(in+out)", f"D:/OCSVM/without_geometric_features/validation(in+out)/numpy/{i}/", i)
                 
class extractandTrain:
    def extract_train(self,directory, outlier_type):
        error_files = []
        self.ocsvm = SGDOneClassSVM()
        las_files = [os.path.join(directory, f) for f in os.listdir(directory)]
        file2_prefixes = []
        print("starting loop")
        # iterate through las files, run function body on 2 las files with 9 similar characters
        for i, file1_folder in enumerate(las_files):
            folder = os.path.join(directory,file1_folder)
            print(folder)
            for i, file1_path in enumerate(os.listdir(folder)):
                if "outliers" not in file1_path:
                    print(os.path.basename(file1_path))
                    prefix_length = len(os.path.basename(file1_path)[:-4])
                    file1_prefix = os.path.basename(file1_path)[:prefix_length]
                    print(file1_prefix)
                    for file2_path in os.listdir(folder)[i+1:]:
                        file2_prefix = os.path.basename(file2_path)[:prefix_length]
                        file2_prefixes.append(file2_prefixes)
                        if file1_prefix in file2_prefix:
                            print("adding")
                            # read point cloud
                            pc = laspy.read(os.path.join(folder,file1_path))
                            # extract values
                            # X = np.array(pc.x)
                            # Y = np.array(pc.y)
                            # Z = np.array(pc.z)
                            R = np.array(pc.red)
                            G = np.array(pc.green)
                            B = np.array(pc.blue)
                            print("read colours")
                            Intensity = np.array(pc.intensity)
                            print("extracted all features")
                            # load features on a numpy array
                            features = np.c_[R,G,B,Intensity]
                            pc2 = laspy.read(os.path.join(folder,file2_path))
                            # X = np.array(pc2.x)
                            # Y = np.array(pc2.y)
                            # Z = np.array(pc2.z)
                            R = np.array(pc2.red)
                            G = np.array(pc2.green)
                            B = np.array(pc2.blue)
                            print("read colours")
                            intensity = np.array(pc2.intensity) 
                            classification = np.array(pc2.classification)
                            features2 = np.c_[R,G,B,intensity]
                            features2 = features2[classification != outlier_type]
                            features_combined = np.vstack((features,features2))
                            try: 
                                scalar = StandardScaler()
                                scalar.fit_transform(features)
                                
                                print(f"fitting on {file1_prefix}")
                                try:
                                   self.ocsvm.partial_fit(features)
                                except:
                                    error_files.append(os.path.basename(pathname))
                                    pass
                            except:
                                error_files.append(os.path.basename(pathname))
                                pass
        for i, file1_path in enumerate(las_files):
            folder = os.path.join(directory,file1_path)
            for i,file1_path in enumerate(os.listdir(folder)):
                if "outliers" not in file1_path: 
                    prefix_length = len(os.path.basename(file1_path)[:-4])
                    file1_prefix = os.path.basename(file1_path)[:prefix_length]
                    if file1_prefix not in file2_prefixes:
                            print("running function on "+file1_prefix)
                            file1 = laspy.read(os.path.join(folder,file1_path))
                            # X = np.array(file1.x)
                            # Y = np.array(file1.Y)
                            # Z = np.array(file1.Z)
                            R = np.array(file1.red)
                            G = np.array(file1.green)
                            B = np.array(file1.blue)
                            Intensity = np.array(file1.intensity)
                            features = np.c_[R,G,B,Intensity]
                    
                            try: 
                                scalar = StandardScaler()
                                scalar.fit_transform(features)
                                
                                print("fitting on "+file1_path)
                                try:
                                    self.ocsvm.partial_fit(features)                                                                                  
                                    error_files.append(os.path.basename(file1_path))
                                except:
                                    pass
                            except:
                                error_files.append(os.path.basename(file1_path))
                                pass
                    # print("fit on "+pathname)
                    # for i in range(num_batches):
                    #     X_batch = features[i* batch_size: (i+1)* batch_size]
                    #     self.ocsvm.partial_fit(X_batch)
                    #     if (i+1) % 10 == 0: 
                    #         print("Trained %d batches" % (i+1))
        return error_files



 



def ocsvmpredict(dir_path,out_path):
        error_files = []
        pathnames = []
        for filename in os.listdir(dir_path):
            if not os.path.isdir(filename):
                if filename.endswith(".laz") and not filename.endswith(("_outliers.laz", "outliers.las")):
                    path = os.path.join(dir_path,filename).replace("//","/")
                    pathnames.append(path)
    
        # iterate over the point clouds and return an 
        for pathname in pathnames:
            # read point cloud
            pc = laspy.read(pathname)
            print(f"reading {pathname}")
            # extract values
            # X = np.array(pc.x)
            # Y = np.array(pc.y)
            # Z= np.array(pc.z)
            R = np.array(pc.red)
            G = np.array(pc.green)
            B = np.array(pc.blue)
            print("read colours")
            Intensity = np.array(pc.intensity)
 
            
            
            print("extracted all features")
            # load features on a numpy array
            features = np.c_[R,G,B,Intensity]
          
            outcome = preprocessing.ocsvm.predict(features)
            print("predicted "+pathname)
            # save outcome to .npy format for further analysis
            np.save(f"{out_path}{os.path.basename(pathname)[:-4]}",outcome)
          
             #create new las file, append outcome as classification to las file 
            # new_file = laspy.create(point_format=pc.header.point_format,file_version=pc.header.version)
            # new_file.points = pc.points
            # new_file.point_source_id = np.asarray(outcome)
            # new_file.outliers= np.asarray(outcome)
            # # save lasfile to specified directory using the input name
            # filename = out_path+os.path.basename(pathname)[:-4]+"_ocsvm.las"
            # if not os.path.exists(os.path.join(dir_path, filename)):
            #     new_file.write(f"{out_path}{os.path.basename(pathname)[:-4]}_ocsvm.las")
        return error_files

# dump(preprocessing.ocsvm,"model_joblib")
# print(preprocessing.ocsvm)
preprocessing = extractandTrain()
features = [0,1,2,3]
for i in features:
    preprocessing.extract_train("D:/OCSVM/without_geometric_features/train+outlier_type/",i)
    ocsvmpredict("D:/OCSVM/without_geometric_features/test/",f"D:/OCSVM/without_geometric_features/test/numpy/{i}/")

input_dir = 
input_las_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith((".laz",".las"))]
outlier_las_files = [os.path.join(outlier_dir, f) for f in os.listdir(outlier_dir) if f.endswith((".laz",".las"))]
file2_prefixes = []
for i, file1_path in enumerate(input_las_files):
 
    if os.path.isdir(os.path.join(input_dir,file1_path)):
        folder = os.path.join(input_dir,file1_path)
        for file1_path in os.listdir(folder):
            prefix_length = len(os.path.basename(file1_path)[:-4])
            file1_prefix = os.path.basename(file1_path)[:prefix_length]
    else:
        prefix_length = len(os.path.basename(file1_path)[:-4])
        file1_prefix = os.path.basename(file1_path)[:prefix_length]
        
    for file2_path in outlier_las_files:
        
        if os.path.isdir(os.path.join(outlier_dir,file2_path)):
            folder = os.path.join(input_dir,folder)
            for file2_path in folder:
                file2_prefix = os.path.basename(file2_path)[:prefix_length]
                file2_prefixes.append(file2_prefixes)
            
        if file1_prefix in file2_prefix:
            if filename.endswith(".laz") and not "nocolor" in filename:
                path = os.path.join(folder,filename).replace("//","/")
                self.pathnames.append(path)
                
    else:
        for filename in os.listdir(directory):
            if filename.endswith(".laz") and not "nocolor" in filename:
                path = os.path.join(directory,filename).replace("//","/")
                if path not in self.pathnames:
                    self.pathnames.append(path)