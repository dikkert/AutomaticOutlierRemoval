from sklearn.metrics import accuracy_score, precision_score, recall_score 
import numpy as np
import os

def testAccuracy(input_path,prefix_length=9):
    total_acc = []
     # create lists of input paths
    las_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".npy")]
    # check for similar files
    for i, file1_path in enumerate(las_files):
        file1_prefix = os.path.basename(file1_path)[:prefix_length]
        for file2_path in las_files[i+1:]:
            file2_prefix = os.path.basename(file2_path)[:prefix_length]
            if file1_prefix == file2_prefix:
                print(f"Running function on {file1_path} and {file2_path}")
                with np.load(file1_path) and np.load(file2_path) as train, test: 
                    acc = accuracy_score(train[:,-1],test[:-1])
                    total_acc.append(acc)
    print(total_acc)
    return total_acc

