from sklearn.metrics import accuracy_score, precision_score, recall_score 
import numpy as np

def testAccuracy(self, input_path1, input_path2)
     # create lists of input paths
    las_files1 = [os.path.join(input_path1, f) for f in os.listdir(input_path2) if f.endswith(".npy")]
    las_files2 = [os.path.join(input_path2, f) for f in os.listdir(input_path2) if f.endswith(".npy")]
    # check for similar files
    for i, file1_path in enumerate(las_files):
    file1_prefix = os.path.basename(file1_path)
    for file2_path in las_files[i+1:]:
        file2_prefix = os.path.basename(file2_path)
        if file1_prefix == file2_prefix:
            print(f"Running function on {file1_path} and {file2_path}"):
            with np.load(file1_path) and np.load(file2_path) as train, test
            acc = accuracy_score(train[:,-1],test[:-1])

# Example predicted and true labels
y_pred = [0, 1, 0, 0, 1, 1, 1]
y_true = [0, 0, 1, 0, 1, 0, 1]

# Compute accuracy score
accuracy = accuracy_score(y_true, y_pred)

# Print the accuracy score
print(f"Accuracy score: {accuracy:.4f}")