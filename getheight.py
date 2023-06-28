# -*- coding: utf-8 -*
"""
Created on Thu Mar 30 11:22:09 2023

@author: Dick
"""

import laspy

import numpy
import matplotlib.pyplot as plt
from owslib.wcs import WebCoverageService

import rasterio
from rasterio.plot import show
import os

# open las file
lasfile = laspy.read("D:/height_project/data/nrd/tunnel_noord_4_translate_SLOPE_2_500_0.5_Ground.las")
print(list(lasfile.point_format.dimension_names))
# extract xyz values
x,y,z = lasfile.x, lasfile.y, lasfile.z
# extract bbox
xmin = lasfile.header.min[0]
xmax = lasfile.header.max[0]
ymin = lasfile.header.min[1]
ymax = lasfile.header.max[1]
bbox = (xmin-100, ymin-100, xmax+100, ymax+100)





# Access the WCS by proving the url and optional arguments
wcs = WebCoverageService("https://service.pdok.nl/rws/ahn3/wcs/v1_0?&service=wcs&request=GetCapabilities", version='1.0.0')
# Request the DSM data from the WCS
response = wcs.getCoverage(identifier='ahn3_05m_dtm', bbox=bbox, format='GEOTIFF_FLOAT32', crs='urn:ogc:def:crs:EPSG::28992', resx=0.5, resy=0.5)

# Write the data to a local file in the 'data' directory
with open('data/AHN3_05m_DSM.tif', 'wb') as file:
    file.write(response.read())
    
# Open the raster
dsm = rasterio.open("data/AHN3_05m_DSM.tif", driver="GTiff")


# Plot with rasterio.plot, which provides Matplotlib functionality
plt.figure(figsize=(5, 5), dpi=300)  # adjust size and resolution
show(dsm, title='Digital Surface Model', cmap='gist_ncar')


# Define the height threshold as a difference in meters
height_threshold = 1.0


# Open the raster file and get the elevation values
elevation_values = dsm.read(1)
print(elevation_values[:10])
print(elevation_values.shape)
print(elevation_values[0,0])
# Get the resolution of the raster file
resolution = dsm.res[0]
print(z[:10])

# Loop through the LAS points and remove those that are outside the heigh{t threshold
for i in range(len(z)):
    try: 
        # Get the X and Y coordinate values of the LAS point
        x = lasfile.x[i]
        y = lasfile.y[i]
    
        # Convert the X and Y coordinate values to row and column indices in the raster file
        row, col = dsm.index(x, y)
    
        # Get the elevation value at the LAS point's location in the raster file
        print(row,col)
        elevation = elevation_values[row, col]
    
        # Calculate the difference between the LAS point's elevation and the raster elevation
        difference = abs(z[i] - elevation)
        # Check if the difference exceeds the height threshold
        if difference != 3.4028234663852886e+38:
            if difference > height_threshold:
                print(difference)
                # Remove the LAS point by setting its Z coordinate to NaN
                lasfile.classification[i] = 1
    except:
        pass

lasfile.points[lasfile.classification != 1]
outliers = laspy.LasData(lasfile.header)
outliers.points = lasfile.points[lasfile.classification == 1]
outliers.write("data/")
lasfile.write("data/Benelux_tunnel_1_translate_SLOPE_2_500_0.5_Ground_erro_3.laz")
# Save the modified LAS file
lasfile.close()

for i in range(len(z)):
    print(i)
for i in range(len(z)):
   if i == 10:
       break
   row, col = dsm.index(lasfile.x[i],lasfile.y[i])
   print(row,col)

elevation_values.shape
       


def compute_differences_AHN(directory,height_threshold):
    # structure for iterative file handling
    pathnames = []
    for filename in os.listdir(directory):
        if filename.endswith((".laz",".las")):
            path = os.path.join(directory,filename).replace("\\","/")
            if path not in pathnames:
                pathnames.append(path)
        # iterate over the point clouds and return an 
    for pathname in pathnames:
        # filename for further saving
        filename = os.path.basename(pathname)
        # open las file
        lasfile = laspy.read(pathname)
        # extract xyz values
        x,y,z = lasfile.x, lasfile.y, lasfile.z
        # extract bbox
        xmin = lasfile.header.min[0]
        xmax = lasfile.header.max[0]
        ymin = lasfile.header.min[1]
        ymax = lasfile.header.max[1]
        bbox = (xmin-100, ymin-100, xmax+100, ymax+100)
        
        # Access the WCS by proving the url and optional arguments
        wcs = WebCoverageService("https://service.pdok.nl/rws/ahn3/wcs/v1_0?&service=wcs&request=GetCapabilities", version='1.0.0')
        # Request the DSM data from the WCS
        response = wcs.getCoverage(identifier='ahn3_05m_dtm', bbox=bbox, format='GEOTIFF_FLOAT32', crs='urn:ogc:def:crs:EPSG::28992', resx=0.5, resy=0.5)

        # Write the data to a local file in the 'data' directory
        with open("data/"+filename+".tif", 'wb') as file:
            file.write(response.read())
            
        # Open the raster
        dsm = rasterio.open("data/"+filename+".tif", driver="GTiff")

        # # Plot with rasterio.plot, which provides Matplotlib functionality
        # plt.figure(figsize=(5, 5), dpi=300)  # adjust size and resolution
        # show(dsm, title='Digital Surface Model', cmap='gist_ncar')

        # Open the raster file and get the elevation values
        elevation_values = dsm.read(1)
        print("working on"+ filename)
        # Loop through the LAS points and remove those that are outside the heigh{t threshold
        for i in range(len(z)):
            try: 
                # Get the X and Y coordinate values of the LAS point
                x = lasfile.x[i]
                y = lasfile.y[i]
            
                # Convert the X and Y coordinate values to row and column indices in the raster file
                row, col = dsm.index(x, y)
            
                # Get the elevation value at the LAS point's location in the raster file
                elevation = elevation_values[row, col]
            
                # Calculate the difference between the LAS point's elevation and the raster elevation
                difference = abs(z[i] - elevation)
                # Check if the difference exceeds the height threshold
                if difference != 3.4028234663852886e+38:
                    if difference > height_threshold:
                        print(difference)
                        # Remove the LAS point by setting its Z coordinate to NaN
                        lasfile.classification[i] = 1
            except:
                pass
        
        lasfile.write("data/results/"+filename[:-4]+"_class.laz")
        print("finished with"+filename)
compute_differences_AHN("D:/height_project/data/nrd", 5.0)

