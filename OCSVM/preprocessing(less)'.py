from sklearn.linear_model import SGDOneClassSVM 
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import laspy
import joblib
from joblib import dump
from  jakteristics import FEATURE_NAMES


def normalize_array(input):
    norm = np.linalg.norm(input)
    if norm == 0:
        return input
    return input / norm


# Loop over each LAS file and concatenate its points with the points for the corresponding prefix
def PrepareOcsvmTestData(directory, output_path):
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
                        X = np.array(file1.x)
                        any_nan = np.isnan(X).any()
                        if any_nan is True:
                            print("some of x is Nan")
                            break
                        Y = np.array(file1.y)
                        Z = np.array(file1.z)
                        R = np.array(file1.red)
                        G = np.array(file1.green)
                        B = np.array(file1.blue)
                        Intensity = np.array(file1.intensity)
                        features1 = np.c_[R,G,B,Intensity]
                        file2 = laspy.read(file2_path)
                        X = np.array(file2.x)
                        Y = np.array(file2.y)
                        Z = np.array(file2.Z)
                        R = np.array(file2.red)
                        G = np.array(file2.green)
                        B = np.array(file2.blue)
                        Intensity = np.array(file2.intensity)
                        features2 = np.c_[R,G,B,Intensity]
                        combined_points = np.vstack((features1, features2))
                        # hstack 
                        # c_ 
                        print("combine points")
                    
                        combined_source_ids = np.concatenate((np.ones((len(file1.points),),dtype=int), np.full(len((file2.points),),-1, dtype=int)))
                        combined_full = np.c_[combined_points,combined_source_ids]
                        # write to binary
                        
                        np.save(f"{output_path}{file1_prefix}_outliers",combined_source_ids)
                        
    for i, file1_path in enumerate(las_files):
        if "outliers" not in file1_path: 
            prefix_length = len(os.path.basename(file1_path)[:-4])
            file1_prefix = os.path.basename(file1_path)[:prefix_length]
            if file1_prefix not in file2_prefixes:
                filename = output_path+file1_prefix+"_outliers.npy"
                if not os.path.exists(os.path.join(directory, filename)):
                    print("running function on "+file1_prefix)
                    file1 = laspy.read(file1_path)
                    X = np.array(file1.x)
                    Y = np.array(file1.Y)
                    Z = np.array(file1.Z)
                    R = np.array(file1.red)
                    G = np.array(file1.green)
                    B = np.array(file1.blue)
                    Intensity = np.array(file1.intensity)
                    features1 = np.c_[R,G,B,Intensity]
                    combined_source_ids = np.ones((len(file1.points),),dtype=int)
                    # write to binary
                    np.save(f"{output_path}{file1_prefix}_outliers(nochange)",combined_source_ids)
            

class extractandTrain:
    def extract_train(self,directory,batch_size=10000):
        error_files = []
        self.all_files = np.empty((0,7))
        self.ocsvm= SGDOneClassSVM()
        # creates a list of paths to files to read the point clouds
        self.pathnames = []
        for folder in os.listdir(directory):
            if os.path.isdir(os.path.join(directory,folder)):
                folder = os.path.join(directory,folder)
                for filename in os.listdir(folder):
                    if filename.endswith(".laz") and not "nocolor" in filename:
                        path = os.path.join(folder,filename).replace("//","/")
                        self.pathnames.append(path)
                        
            else:
                for filename in os.listdir(directory):
                    if filename.endswith(".laz") and not "nocolor" in filename:
                        path = os.path.join(directory,filename).replace("//","/")
                        if path not in self.pathnames:
                            self.pathnames.append(path)
                            
        print(len(self.pathnames))
            # iterate over the point clouds and return an 
        for pathname in self.pathnames:
            # read point cloud
            pc = laspy.read(pathname)
            print("reading {pathname}")
            # extract values
            X = np.array(pc.x)
            Y = np.array(pc.y)
            Z = np.array(pc.z)
            R = np.array(pc.red)
            G = np.array(pc.green)
            B = np.array(pc.blue)
            print("read colours")
            Intensity = np.array(pc.intensity)
            eigenvalues_sum = np.array(pc['eigenvalue_sum'])
            omnivariance = np.array(pc['omnivariance'])
            eigentropy = np.array(pc['eigenentropy'])
            anisotropy = np.array(pc['anisotropy'])
            planarity = np.array(pc['planarity'])
            linearity = np.array(pc['linearity'])
            pca1 = np.array(pc['PCA1'])
            pca2 = np.array(pc['PCA2'])
            surface_variation = np.array(pc['surface_variation'])
            sphericity = np.array(pc['sphericity'])
            verticality = np.array(pc['verticality'])
            nx = np.array(pc["nx"])
            ny = np.array(pc["ny"])
            nz = np.array(pc["nz"])
            number_of_neighbors = np.array(pc['number_of_neighbors'])
            eigenvalue1 = np.array(pc['eigenvalue1'])
            eigenvalue2 = np.array(pc['eigenvalue2'])
            eigenvalue3 = np.array(pc['eigenvalue3'])
            eigenvector1x = np.array(pc['eigenvector1x'])
            eigenvector1y = np.array(pc['eigenvector1y'])
            eigenvector1z = np.array(pc['eigenvector1z'])
            eigenvector2x = np.array(pc['eigenvector2x'])
            eigenvector2y = np.array(pc['eigenvector2y'])
            eigenvector2z = np.array(pc['eigenvector2z'])
            eigenvector3x = np.array(pc['eigenvector3x'])
            eigenvector3y = np.array(pc['eigenvector3y'])
            eigenvector3z = np.array(pc['eigenvector3z'])
            print("extracted all features")
            # load features on a numpy array
            features = np.c_[X,Y,Z,R,G,B,Intensity,eigenvalues_sum,omnivariance,eigentropy,anisotropy,pca2,surface_variation,number_of_neighbors,eigenvalue3]
            try: 
                scalar = StandardScaler()
                scalar.fit_transform(features)
                
                print("fitting on "+pathname)
                try:
                    self.ocsvm.partial_fit(features)
                except:
                    error_files.append(os.path.basename(pathname))
                    pass
            except:
                error_files.append(os.path.basename(pathname))
                pass
            # print("fit on "+pathname)
            # for i in range(num_batches):
            #     X_batch = features[i* batch_size: (i+1)* batch_size]
            #     self.ocsvm.partial_fit(X_batch)
            #     if (i+1) % 10 == 0: 
            #         print("Trained %d batches" % (i+1))
        dump(self.ocsvm, "ocsvm_model.joblib")
        return error_files

    

    def ocsvmpredict(self,dir_path,out_path):
        self.pathnames = []
        for filename in os.listdir(dir_path):
            if not os.path.isdir(filename):
                if filename.endswith((".laz",",.las")) and not filename.endswith(("_outliers.laz", "outliers.las")):
                    path = os.path.join(dir_path,filename).replace("//","/")
                    self.pathnames.append(path)
    
        # iterate over the point clouds and return an 
        for pathname in self.pathnames:
            # read point cloud
            pc = laspy.read(pathname)

            # extract values
            X = np.array(pc.x)
            Y = np.array(pc.y)
            Z = np.array(pc.z) 
            R = np.array(pc.red)
            G = np.array(pc.green)
            B = np.array(pc.blue)
            Intensity = np.array(pc.intensity)
            point_ID = np.array(pc.point_source_id)
            return_number = np.array(pc.return_number)
            nr_of_returns = np.array(pc.number_of_returns)
            normalize_array(X)
            normalize_array(Y)
            normalize_array(Z)
           # load features on a numpy array
            features = np.c_[R,G,B,Intensity]
            # run prediction using traied ocsvm model on features
            outcome = self.ocsvm.predict(features)
            # save outcome to .npy format for further analysis
            np.savez_compressed(f"{out_path}{os.path.basename(pathname)[:-4]}",outcome)
            
            # create new las file, append outcome as classification to las file 
            new_file = laspy.create(point_format=pc.header.point_format,file_version=pc.header.version)
            new_file.points = pc.points
            new_file.add_extra_dim(laspy.ExtraBytesParams(
                name="outlier",
                type="3f8",
                description="outlier classification, 1 is inlier -1 is outlier"
            ))
            new_file.outliers= np.asarray(outcome)
            # save lasfile to specified directory using the input name
            new_file.write(f"{out_path}{os.path.basename(pathname)[:-4]}_ocsvm.laz")
        return 

        
# PrepareOcsvmTestData("D:/OCSVM/validation(in+out)/","D:/OCSVM/validation(in+out)/numpy/")
# pc = laspy.read("D:/OCSVM/train/BET_OB_VL_N/fixed/BET-VL-AB-028_2023-03-13_13h23_31_254.las")

preprocessing = extractandTrain()
preprocessing.extract_train("D:/OCSVM/with_geometric_features/train")




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
            print("reading {pathname}")
            # extract values
            X = np.array(pc.x)
            Y = np.array(pc.y)
            Z= np.array(pc.z)
            R = np.array(pc.red)
            G = np.array(pc.green)
            B = np.array(pc.blue)
            print("read colours")
            Intensity = np.array(pc.intensity)
 
            eigenvalues_sum = np.array(pc['eigenvalue_sum'])
            omnivariance = np.array(pc['omnivariance'])
            eigentropy = np.array(pc['eigenentropy'])
            anisotropy = np.array(pc['anisotropy'])
            #planarity = np.array(pc['planarity'])
            #linearity = np.array(pc['linearity'])
            #pca1 = np.array(pc['PCA1'])
            pca2 = np.array(pc['PCA2'])
            surface_variation = np.array(pc['surface_variation'])
            #sphericity = np.array(pc['sphericity'])
            #verticality = np.array(pc['verticality'])
            #nx = np.array(pc['nx'])
            #ny = np.array(pc['ny'])
            #nz = np.array(pc['nz'])
            number_of_neighbors = np.array(pc['number_of_neighbors'])
            #eigenvalue1 = np.array(pc['eigenvalue1'])
            #eigenvalue2 = np.array(pc['eigenvalue2'])
            eigenvalue3 = np.array(pc['eigenvalue3'])
            #eigenvector1x = np.array(pc['eigenvector1x'])
            #eigenvector1y = np.array(pc['eigenvector1y'])
            #eigenvector1z = np.array(pc['eigenvector1z'])
            #eigenvector2x = np.array(pc['eigenvector2x'])
            #eigenvector2y = np.array(pc['eigenvector2y'])
            #eigenvector2z = np.array(pc['eigenvector2z'])
            #eigenvector3x = np.array(pc['eigenvector3x'])
            #eigenvector3y = np.array(pc['eigenvector3y'])
            #eigenvector3z = np.array(pc['eigenvector3z'])
            
            print("extracted all features")
            # load features on a numpy array
            features = np.c_[X,Y,Z,R,G,B,Intensity,eigenvalues_sum,omnivariance,eigentropy,anisotropy,pca2,surface_variation,number_of_neighbors,eigenvalue3]
            try:
                outcome = preprocessing.ocsvm.predict(features)
            except:
                error_files.append(os.path.basename(pathname))
                pass
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

# dump(preprocessing.ocsvm,"model_joblib")
# print(preprocessing.ocsvm)
ocsvmpredict("D:/OCSVM/with_geometric_features/test/","D:/OCSVM/with_geometric_features/test/numpy/")

pc = laspy.read("D:/OCSVM/test/BET_NB_GN_0-034_2023-03-14_00h06_04_664.las")
print(list(pc.point_format.dimension_names))
eigen1 = np.array(pc["Volume__density__(r_eq_0.05)"])
print(eigen1[:100])


import csv
import matplotlib.pyplot as plt
ocsvmpredict("D:/OCSVM/test", "D:/OCSVM/test/numpy/")
values = preprocessing.ocsvm.coef_
print(values)
keys = features = ["R","G","B","intensity","density"]
my_dict = dict(zip(original_values,values))
with open('coef_density_large_train_2.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['Key', 'Value'])
    for key, value in my_dict.items():
        writer.writerow([key, value])
plt.bar(original_values, values)
plt.show()

original_values = ["X", "Y", "Z", "R", "G", "B", "Intensity", "eigenvalues_sum", "omnivariance", "eigentropy", "anisotropy", "pca2", "surface_variation", "number_of_neighbors", "eigenvalue3"]
string_values = [str(value) for value in original_values]
print(my_dict)
print(FEATURE_NAMES)
keys  = [
    "X","Y", "Z","R", "G", "B", "Intensity", "eigenvalues_sum", "omnivariance", "eigentropy", 
    "anisotropy", "planarity", "linearity", "pca1", "pca2", "surface_variation", 
    "sphericity", "verticality", "number_of_neighbors", "eigenvalue1", "eigenvalue2", 
    "eigenvalue3", "eigenvector1x", "eigenvector1y", "eigenvector1z", "eigenvector2x", 
    "eigenvector2y", "eigenvector2z", "eigenvector3x", "eigenvector3y", "eigenvector3z"
]
print(len(keys))
print(selected_features)

        

 
