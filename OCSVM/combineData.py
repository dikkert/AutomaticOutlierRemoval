import numpy as np
import laspy
import os
import pandas as pd


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
                print("read files")
                print(len(file1.points))
                combined_points = np.vstack((features1, features2))
                # hstack 
                # c_ 
                print("combine points")
            
                combined_source_ids = np.concatenate((np.ones((len(file1.points),),dtype=int), np.full(len((file2.points),),-1, dtype=int)))
                combined_full = np.c_[combined_points,combined_source_ids]
                np.save(output_path+os.path.basename(file1_path),combined_source_ids)
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
                # write to binary 
               
                
#print(list(file1.point_format.dimension_names))
PrepareOcsvmTestData("D:/beneluxtunnel/buisD/","D:/beneluxtunnel/buisD/test/test_output/")

# Read in the first file
file1 = laspy.read("D:/beneluxtunnel/buisD/test/BET-D-008.las")
print(len(file1.points))
X = np.array(file1.x)
print(len(X))
Y = np.array(file1.Y)
Z = np.array(file1.Z)
R = np.array(file1.red)
G = np.array(file1.green)
B = np.array(file1.blue)
Intensity = np.array(file1.intensity)
features1 = np.c_[X,Y,Z,R,G,B,Intensity]
new_file = laspy.create(point_format=file1.header.point_format)

# Read in the second file
file2 = laspy.read("D:/beneluxtunnel/buisD/test/BET-D-008_outlier.las")
print(len(file2.points))
X = np.array(file2.x)
Y = np.array(file2.y)
Z = np.array(file2.Z)
R = np.array(file2.red)
G = np.array(file2.green)
B = np.array(file2.blue)
Intensity = np.array(file2.intensity)
features2 = np.c_[X,Y,Z,R,G,B,Intensity]
print(features2)

combined_points = np.vstack((features1, features2))

print(len(features2))
print(len(combined_points))
print(combined_points[:1])
print("combine points")
            
combined_source_ids = np.concatenate((np.zeros((len(file1.points),),dtype=int), np.ones(len((file2.points),), dtype=int)))
combined_full = np.c_[combined_points,combined_source_ids]
combined_full.shape
print(len(combined_full[:,0]))
output_file = laspy.create(point_format=file1.header.point_format)
                # write numpy array data back to .las
output_file.x = combined_points[:,0]
output_file.y = combined_points[:,1]
output_file.z = combined_points[:,2]
output_file.red = combined_points[:,3]
output_file.green = combined_points[:,4]
output_file.blue = combined_points[:,5]
output_file.intensity = combined_points[:,6]
output_file.classsification = combined_full[:,7]
output_file.write("D:/beneluxtunnel/buisD/test/test_output/BET-D-008.las")
merged_file = laspy.read("D:/beneluxtunnel/buisD/test/test_output/BET-D-008.las")
print(len(merged_file.points))
combined_points = np.concatenate(features1, features2)


# show tabular
df = pd.DataFrame(combined_full)
print(df[:-10])

