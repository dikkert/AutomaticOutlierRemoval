from sklearn.linear_model import SGDOneClassSVM
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import laspy


def normalize_array(input):
    norm = np.linalg.norm(input)
    if norm == 0:
        return input
    return input / norm
    
# Loop over each LAS file and concatenate its points with the points for the corresponding prefix
def PrepareOcsvmTestData(directory, output_path):
    '''function that reads all .las files in a directory and returns a classified by source ID .las file containing '''
    # list all .las files
    las_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".las")]
    prefix_length= 9
    # iterate through las files, run function body on 2 las files with 9 similar characters
    for i, file1_path in enumerate(las_files):
        file1_prefix = os.path.basename(file1_path)[:prefix_length]
        for file2_path in las_files[i+1:]:
            file2_prefix = os.path.basename(file2_path)[:prefix_length]
            if file1_prefix == file2_prefix:
                print(f"Running function on {file1_path} and {file2_path}")
    # function body, read las files, concatenate points, write new .las file containing combined points, classified with source IDs
                # read and extract all file features for file 1 and file 2
                file1 = laspy.read(file1_path)
                X = np.array(file1.x)
                Y = np.array(file1.Y)
                Z = np.array(file1.Z)
                R = np.array(file1.red)
                G = np.array(file1.green)
                B = np.array(file1.blue)
                Intensity = np.array(file1.intensity)
                features1 = np.c_[X,Y,Z,R,G,B,Intensity]
                file2 = laspy.read(file2_path)
                X = np.array(file2.x)
                Y = np.array(file2.y)
                Z = np.array(file2.Z)
                R = np.array(file2.red)
                G = np.array(file2.green)
                B = np.array(file2.blue)
                Intensity = np.array(file2.intensity)
                features2 = np.c_[X,Y,Z,R,G,B,Intensity]
                combined_points = np.vstack((features1, features2))
                # hstack 
                # c_ 
                print("combine points")
            
                combined_source_ids = np.concatenate((np.zeros((len(file1.points),),dtype=int), np.ones(len((file2.points),), dtype=int)))
                combined_full = np.c_[combined_points,combined_source_ids]
                 # write to binary
                np.save(output_path+os.path.basename(file2_path),combined_full)
                # output_file = laspy.create(point_format=file1.header.point_format)
                # # write numpy array data back to .las
                # output_file.x = combined_points[:,0]
                # output_file.y = combined_points[:,1]
                # output_file.z = combined_points[:,2]
                # output_file.red = combined_points[:,3]
                # output_file.green = combined_points[:,4]
                # output_file.blue = combined_points[:,5]
                # output_file.intensity = combined_points[:,6]
                # output_file.classsification = combined_full[:,7]
                # output_file.write(output_path+os.path.basename(file1_path))
                # print("done writing")
                


class preprocessing:
    def extract_pc(self,directory):
        '''function that reads a point cloud and returns a numpy array containing point ID, intensity '''
        # create empty array for the collection of arrays
        self.all_files = np.empty((0,7))
        # creates a list of paths to files to read the point clouds
        self.pathnames = []
        for filename in os.listdir(directory):
            path = os.path.join(directory,filename)
            self.pathnames.append(path)
    
        # iterate over the point clouds and return an 
        for pathname in self.pathnames:
            # read point cloud
            pc = laspy.read(pathname)

            # extract values
            X = np.array(pc.x)
            Y = np.array(pc.Y)
            Z = np.array(pc.Z) 
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
            features = np.c_[X,Y,Z,R,G,B,Intensity]
            self.all_files = np.vstack((self.all_files,features))
            print(self.all_files.shape)
        return self.all_files

    def ocsvmtrainer(self):
        # normalize all values
        scalar = StandardScaler()
        scalar.fit_transform(self.all_files)
        # define the method
        self.ocsvm= SGDOneClassSVM()
        # iterate the ocsvm to partially fit the training data
        batch_size = 10000
        num_batches= self.all_files.shape[0] // batch_size
        for i in range(num_batches):
            X_batch = self.all_files[i* batch_size: (i+1)* batch_size]
            self.ocsvm.partial_fit(X_batch)
            if (i+1) % 10 == 0: 
                print("Trained %d batches" % (i+1))
                #self.ocsvm.fit(scalar.inverse_transform(sgd.coef_))

    def ocsvmpredict(self,dir_path):
        self.pathnames = []
        for filename in os.listdir(dir_path):
            path = os.path.join(dir_path,filename)
            self.pathnames.append(path)
    
        # iterate over the point clouds and return an 
        for pathname in self.pathnames:
            # read point cloud
            pc = laspy.read(pathname)

            # extract values
            X = np.array(pc.x)
            Y = np.array(pc.Y)
            Z = np.array(pc.Z) 
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
            self.features = np.c_[X,Y,Z,R,G,B,Intensity]
            outcome = self.ocsvm.predict(self.features)
            outcome_file = np.c_[self.all_files,outcome]
            np.save(output_path+os.path.basename(pathname),outcome_file)
            print(outcome_file.shape)
        return outcome_file
preprocessing = preprocessing()
preprocessing.extract_pc("D:/beneluxtunnel/diensttunnels/BET_NB_GZ_0/output/")
preprocessing.ocsvmtrainer()
preprocessing.extract_pc("D:/beneluxtunnel/diensttunnels/BET_NB_GZ_0/test")
preprocessing.ocsvmpredict()


