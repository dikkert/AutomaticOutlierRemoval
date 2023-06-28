# AutomaticOutlierRemoval
This repository is created as a part of the MGI Internship Statistical Learning for Point Cloud Feature Extraction

## OCSVM 
to run the OCSVM model go to OCSVM/preprocessing(less)
select the features to use ( recommended: RGB and Intensity)
set the ExtractandTrainClass and run on training data (included in code), it's advisable to select training data which is similar to the type of structure processed (tunnels for tunnels, rooms for rooms)
run ocsvmpredict on your own data. uncomment the lasfile saving section at the end to save the result as a lasfile

## getheight
run this script to classify all points with a height difference of 5 meter from the AHN (slow script)

## remove_Whites
run this script to remove all fully white points (Almost no inliers are classified as fully white)

