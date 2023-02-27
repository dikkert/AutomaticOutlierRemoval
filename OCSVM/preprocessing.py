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

    def ocsvmpredict(self,directory):
        outcome = self.ocsvm.predict(self.all_files)
        outcome_file = np.c_[self.all_files,outcome]
        print(outcome_file.shape)
        return outcome_file
preprocessing = preprocessing()
preprocessing.extract_pc("D:/beneluxtunnel/diensttunnels/BET_NB_GZ_0/output/")
preprocessing.ocsvmtrainer()
preprocessing.extract_pc("D:/beneluxtunnel/diensttunnels/BET_NB_GZ_0/test")
preprocessing.ocsvmpredict()


