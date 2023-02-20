# import packages
import laspy
import numpy as np 
import re
import os

# if you want to check more dimensions..
#list(pc.point_format.dimension_names)

def extract_pc(directory):
    '''function that reads a point cloud and returns a numpy array containing point ID, intensity '''
    # create empty array for the collection of arrays
    all_files = []
    # creates a list of paths to files to read the point clouds
    pathnames = []
    for filename in os.listdir(directory):
        path = os.path.join(directory,filename)
        pathnames.append(path)
    
    # iterate over the point clouds and return an 
    for pathname in pathnames:
        # read point cloud
        pc = laspy.read(pathname)

        # extract values 
        R = pc.red
        G = pc.green
        B = pc.blue
        Intensity = pc.intensity
        point_ID = pc.point_source_id
        return_number = pc.return_number
        nr_of_returns = pc.number_of_returns

        # load features on a numpy array
        features = [R,G,B,Intensity]

        all_files.append(features)
    return all_files


directory = "D:/beneluxtunnel/diensttunnels/BET_NB_GZ_0/output/" 
extract_pc(directory)

print(all_files.shape)
#print(all_files[:10])




    
