import os

def compute_features(input_path,cloudcomparepath='"C:/Program Files/CloudCompare_v2.13.alpha_bin_x64/CloudCompare.exe"'): 

    list_features = ["SUM_OF_EIGENVALUES","ANISOTROPY", "PLANARITY", "LINEARITY","SURFACE_VARIATION"]
    # loop over all files in the directory
    for filename in os.listdir(input_path):
        # check if the file is a point cloud
        if filename.endswith('.las') or filename.endswith('.laz'):
            for feature in list_features: 
                
                # set the input and output filenames
                input_file = os.path.join(input_path, filename)
            
                # construct the command to compute geometric features
                command = cloudcomparepath+" -SILENT -O {} -FEATURE {} {} -FWF_SAVE_CLOUDS {}".format(input_file,feature, 0.01, input_file)
                print(command)
                
                # execute the command
                os.system(command)
                command = cloudcomparepath+" -SILENT -O {} -DENSITY {} -FWF_SAVE_CLOUDS {}".format(input_file, 0.01, input_file)
                os.system(command)

# compute_features("D:/OCSVM/train/BET_OB_VL_N/fixed")

def compute_features2(input_path,cloudcomparepath='"C:/Program Files/CloudCompare_v2.13.alpha_bin_x64/CloudCompare.exe"'): 

    
    list_features =  ["SUM_OF_EIGENVALUES","ANISOTROPY", "PLANARITY", "LINEARITY"]
    # loop over all files in the directory
    for folder in os.listdir(input_path):
            if os.path.isdir(os.path.join(input_path,folder)):
                folder = os.path.join(input_path,folder)
                for filename in os.listdir(folder):
                    if filename.endswith('.laz'):
                        input_file = os.path.join(folder, filename).replace("\\","/")
                        command = cloudcomparepath+"-O {} ".format(input_file)
                        # check if the file is a point cloud
                        # for feature in list_features: 
                        #     command += " -FEATURE {} 0.01".format(feature)
                
                    command += " -DENSITY 0.05 -FWF_SAVE_CLOUDS {}".format(input_file)
                    print(command)
                    os.system(command)

compute_features2("D:/OCSVM/train")

