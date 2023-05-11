# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:15:15 2023

@author: Dick
"""

import faiss
import numpy as np
import laspy

# open las file and extract points
file = laspy.read("D:/OCSVM/without_geometric_features/train/BET_OB_GZ_TR1/BET_OB_GZ_0-043.laz")
x = np.array(file.X).astype("float32")
y = np.array(file.Y).astype("float32")
z = np.array(file.Z).astype("float32")
# combine cooridnates into array
points = np.c_[x,y,z]


# faiss infrastructure
d = 3 # dimensions
nbits = d*2 
nlist =500
quantizer = faiss.IndexLSH(d,nbits) 
index = faiss.IndexIVFFlat(quantizer,d,nlist)
index = faiss.IndexIVFFlat()
print(index.is_trained)
index.train(points)

index = faiss.IndexHNSWFlat(d, 3)

index.add(points)
D,I = index.search(points[:100],6)


#run on gpu ## not possible right now
# res = faiss.StandardGpuResources()
# gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)


index.range_search_L2sqr(x, y, d, nx, ny, radius, result)
D, I = index.range_search(points,0.002)
index.range_search(points,0.2)
print(xb[1])
print(points[1])
