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
def PrepareOcsvmTestData(directory, output_path):
     '''
    This function reads all .las files in a directory and generates a classified .las file containing combined points from multiple files.

    Parameters:
    - directory (str): The path to the directory containing the .las files.
    - output_path (str): The path where the output files will be saved.

    Returns:
    This function doesn't return anything. It generates and saves output files based on the provided input.

    '''
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
        '''
        This method extracts features from point cloud files and trains an One-Class SVM model.

        Parameters:
        - directory (str): The directory containing the point cloud files.
        - batch_size (int): The batch size used for training the model. Default is 10000.

        Returns:
        - error_files (list): A list of filenames for which an error occurred during feature extraction or training.

        '''
        error_files = []
        self.all_files = np.empty((0,4))
        self.ocsvm= SGDOneClassSVM(nu=0.8,warm_start=(True))
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



PrepareOcsvmTestData("D:/OCSVM/without_geometric_features/test/linh/","D:/OCSVM/without_geometric_features/test/linh/numpy/")
# pc = laspy.read("D:/OCSVM/train/BET_OB_VL_N/fixed/BET-VL-AB-028_2023-03-13_13h23_31_254.las")

preprocessing = extractandTrain()
preprocessing.extract_train("D:/OCSVM/without_geometric_features/train")
  
 


def ocsvmpredict(dir_path,out_path):
    '''
    This function predicts outliers in Laz point cloud files using a trained One-Class SVM model.

    Parameters:
    - dir_path (str): The directory path containing the Laz point cloud files.
    - out_path (str): The directory path to save the predicted outliers as `.npy` files.

    Returns:
    - error_files (list): A list of filenames for which an error occurred during the prediction process.

    '''
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
            scalar = StandardScaler()
            scalar.fit_transform(features)
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

#%% if you wish to create a test folder, use this function
PrepareOcsvmTestData("D:/OCSVM/without_geometric_features/test/linh/","D:/OCSVM/without_geometric_features/test/linh/numpy/")



#%% If you want to train the OCSVM model, use the following lines, a dataset has been created and can be found in S:/IntershipDickKuijper. You can predict on your own data
preprocessing = extractandTrain()
preprocessing.extract_train("D:/OCSVM/without_geometric_features/train")
  
ocsvmpredict("D:/OCSVM/without_geometric_features/test/","D:/OCSVM/without_geometric_features/test/numpy/")
