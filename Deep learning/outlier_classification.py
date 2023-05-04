import numpy as np
import laspy
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Load the las file
las = laspy.read("path/to/lasfile.las")

# Extract the features and labels
file1 = laspy.read(file1_path)
X = np.array(file1.x)
Y = np.array(file1.Y)
Z = np.array(file1.Z)
R = np.array(file1.red)
G = np.array(file1.green)
B = np.array(file1.blue)
Intensity = np.array(file1.intensity)
features1 = np.c_[R,G,B,Intensity]
y = np.array(file1.classification)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, alpha=0.0001, solver='adam', random_state=42)
clf.fit(X_train_scaled, y_train)

# Evaluate the model
train_score = clf.score(X_train_scaled, y_train)
test_score = clf.score(X_test_scaled, y_test)
print("Training set score: {:.3f}".format(train_score))
print("Test set score: {:.3f}".format(test_score))

# Predict the labels for the entire las file
X_scaled = scaler.transform(X)
y_pred = clf.predict(X_scaled)

# Save the predicted labels to the las file
las.classification = y_pred
las.write("path/to/lasfile_classified.las")