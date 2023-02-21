from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

from extractdata import extract_pc


# extract data using extract_pc function 
extract_pc("D:/beneluxtunnel/diensttunnels/BET_NB_GZ_0/output/")

scalar = StandardScaler()
X_train = scalar.fit_transform(all_files)

# define the method
ocsvm= OneClassSVM(kernel="rbf", nu=0.1)

## define the SGD classifier 
sgd = SGDClassifier(loss= "hinge", penalty="12", aplha=0.001, random_state=42)

# Train using sgd
batch_size = 100
num_batches= X_train.shape[0] // batch_size
for i in range(num_batches):
    X_batch = X_train[i* batch_size: (i+1)* batch_size]
    sgd.partial_fit(X_batch)
    if (i+1) % 10 == 0:
        print("Trained %d batches" % (i+1))
        ocsvm.fit(scalar.inverse_transform(sgd.coef_))

# save the trained model
#joblib.dump(ocsvm, "ocsvm_model.pkl")