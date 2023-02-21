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
            X = pc.x
            Y = pc.Y
            Z = pc.Z 
            R = pc.red
            G = pc.green
            B = pc.blue
            Intensity = pc.intensity
            point_ID = pc.point_source_id
            return_number = pc.return_number
            nr_of_returns = pc.number_of_returns
            normalize_array(X)
            normalize_array(Y)
            normalize_array(Z)

           # load features on a numpy array
            features = np.column_stack((X,Y,Z,R,G,B,Intensity))
            self.all_files = np.vstack((self.all_files,features))
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

    def predict(self):
        self.ocsvm.predict(self.all_files) 
preprocessing = preprocessing()
preprocessing.extract_pc("D:/beneluxtunnel/diensttunnels/BET_NB_GZ_0/output/")
preprocessing.ocsvmtrainer()
preprocessing.extract_pc("D:/beneluxtunnel/test")
preprocessing.predict()


