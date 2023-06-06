# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:00:13 2023

@author: Dick
"""

"D:/OCSVM/without_geometric_features/validation(in+out)/BET_NB_GN_0-081_outliers.laz",
"D:/OCSVM/without_geometric_features/validation(in+out)/BET_NB_GN_0-035_outliers.laz",
"D:/OCSVM/without_geometric_features/validation(in+out)/BET_NB_GN_0-053_outliers.laz",
"D:/OCSVM/without_geometric_features/validation(in+out)/BET_NB_GN_0-055_outliers.laz",
"D:/OCSVM/without_geometric_features/validation(in+out)/BET_NB_GN_0-057_outliers.laz"
# Loop over each LAS file and concatenate its points with the points for the corresponding prefix
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
                        Y = np.array(file1.y)
                        Z = np.array(file1.z)
                        features1 = np.c_[X,Y,Z]
                        file2 = laspy.read(file2_path)
                        X = np.array(file2.x)
                        Y = np.array(file2.y)
                        Z = np.array(file2.Z)
                       
                        features2 = np.c_[X,Y,Z]
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
            

def compute_all_features(directory,output_dir):
    pathnames = []
    for folder in os.listdir(directory):
        if os.path.isdir(os.path.join(directory,folder)):
            folder = os.path.join(directory,folder)
            for filename in os.listdir(folder):
                if filename.endswith(".npy") filename:
                    path = os.path.join(folder,filename).replace("//","/")
                    pathnames.append(path)
                    
        else:
            for filename in os.listdir(directory):
                if filename.endswith((".npy") in filename:
                    path = os.path.join(directory,filename).replace("//","/")
                    if path not in pathnames:
                        pathnames.append(path)
                    
    
    # iterate over the point clouds and return an 
    
    for pathname in pathnames:
        print(f"opening {pathname}")
        file = np.load(pathname)
        points = file[:,:3]
        
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
            if file[i,4] == -1:
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
        
PrepareOcsvmTestData("D:/OCSVM/without_geometric_features/train","D:/OCSVM/without_geometric_features/train/numpy")
compute_all_features("D:/OCSVM/without_geometric_features/test","D:/OCSVM/with_geometric_features/test")