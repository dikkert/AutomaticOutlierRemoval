import numpy as np
import tensorflow as tf

# creates a list of paths to files to read the point clouds
self.pathnames = []
for folder in os.listdir(directory):
    if os.path.isdir(os.path.join(directory,folder)):
        folder = os.path.join(directory,folder)
        for filename in os.listdir(folder):
            if filename.endswith(".las") and not "nocolor" in filename:
                path = os.path.join(folder,filename).replace("\\","/")
                self.pathnames.append(path)
                
    else:
        for filename in os.listdir(directory):
            if filename.endswith(".las") and not "nocolor" in filename:
                path = os.path.join(directory,filename).replace("\\","/")
                if path not in self.pathnames:
                    self.pathnames.append(path)
                    
    
    # iterate over the point clouds and return an 
for pathname in self.pathnames:
    # read point cloud
    pc = laspy.read(pathname)
    print("reading {pathname}")
    # extract values
    R = np.array(pc.red)
    G = np.array(pc.green)
    B = np.array(pc.blue)
    print("read colours")
    Intensity = np.array(pc.intensity)
    features = np.c_[R,G,B,Intensity]
    totensor = tf.data.Dataset.from_tensor_slices(features)
    # classification
    try:
        # load classes from dataset
        classification = np.array(pc.classification)
        # transform class labels into one hot encoding for example: [0.,0.,1.,0.]
        class_labels = [1,2,3,4]
        class_to_index = {class_label: i for i, class_label in enumerate(class_labels)}
        label_indices = [class_to_index[label] for label in classification]
        one_hot_labels = np.eye(len(class_labels))[label_indices]
        one_hot_label_tensor = tf.constant(one_hot_labels, dtype= tf.float32)
    