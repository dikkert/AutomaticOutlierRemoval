import numpy as np
import laspy
import open3d as o3d
from jakteristics import compute_features, las_utils,FEATURE_NAMES, ckdtree
import math
import laspy

import os


def compute_all_features(directory,output_dir):
    pathnames = []
    for folder in os.listdir(directory):
        if os.path.isdir(os.path.join(directory,folder)):
            folder = os.path.join(directory,folder)
            for filename in os.listdir(folder):
                if filename.endswith((".laz",".las")) and not "nocolor" in filename:
                    path = os.path.join(folder,filename).replace("//","/")
                    pathnames.append(path)
                    
        else:
            for filename in os.listdir(directory):
                if filename.endswith((".laz",".las")) and not "nocolor" in filename:
                    path = os.path.join(directory,filename).replace("//","/")
                    if path not in pathnames:
                        pathnames.append(path)
                    
    
    # iterate over the point clouds and return an 
    
    for pathname in pathnames:
        print(f"opening {pathname}")
        file = laspy.read(pathname)
        x = np.array(file.x)
        y = np.array(file.y)
        z = np.array(file.z)
        points = np.c_[x,y,z]
        
        #create empty list of density values
        densitylist = []

        # # read xyz in o3d format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # build KDTree
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        # set distance
        distance = 0.2
        # iterate over points in point cloud, returning an index list of nearby points
        print("starting density calculation")
        for i, _ in enumerate(pcd.points):
            [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], distance)
            
            # calculate density using the amount of points
            density = len(idx)
            densitylist.append(density)
            if i % 100000 == 0:
                print("point"+str(i))
        print("done with density")
        # transform list to numpy array    
        densityarray = np.asarray(densitylist)

        # compute other geometric features
        #features = compute_features(points,distance,max_k_neighbors=50000,feature_names=FEATURE_NAMES)
        print("done with all other features")
        # add density to list of features
        # FEATURE_NAMES_2 = list(FEATURE_NAMES)+["density"]
        # add density column to array
        # features_and_density = np.c_[features,densityarray]

        directory, filename = os.path.split(pathname)
        train, og_map = os.path.split(directory)
        new_dir = output_dir+"/"+og_map
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        output_path = new_dir+"/"+filename[:-4]+"_features.laz"
        print(f"writing file to {output_path}")
       
        new_file = laspy.create(point_format=file.header.point_format,file_version=file.header.version)
        new_file.points = file.points
        new_file["number_of_neighbors"] = densityarray
        #new_file.density= densityarray()
        # save lasfile to specified directory using the input name
        new_file.write(output_path)
        #las_utils.write_with_extra_dims(pathname, output_path,densityarray,extra_feature)
        
        
compute_all_features("D:/OCSVM/without_geometric_features/test","D:/OCSVM/with_geometric_features/test")
