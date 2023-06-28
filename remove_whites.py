# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:49:45 2023

@author: Dick
"""

import laspy
import numpy as np

# read las file 
in_file = laspy.read("D:/Sijtwende_coloured/tiles/no_white/adj_corr_color_230208_102903_S2.laz")
# load features as int 16
red = in_file.red.astype("uint16")
green = in_file.green.astype("uint16")
blue = in_file.blue.astype("uint16")
x =in_file.X
y = in_file.Y
z = in_file.Z
intensity = in_file.intensity
return_number = in_file.return_number
number_of_returns = in_file.number_of_returns
scan_direction_flag = in_file.scan_direction_flag
scan_angle_rank = in_file.scan_angle_rank

# conditional statement which removes all points which are white (rgb = 65525)
rgb = np.c_[x,y,z,red,green,blue,intensity,return_number,number_of_returns,scan_direction_flag,scan_angle_rank]
np.where((red == 65535)&(green == 65535)&(blue==65535))
indices = np.where((rgb[:,3] != 65535)&(rgb[:,4] !=65535)&(rgb[:,5] != 65535))[0]

print(len(indices))

# Create a new array containing only the non-65535 values
new_arr = rgb[indices]

# write updated points to las file
in_file.points= in_file.points[new_arr]
in_file.write("D:/Sijtwende_coloured/tiles/no_white/adj_corr_color_230208_102903_S2bla.laz")
dimensions = list(in_file.point_format.dimension_names)

new_las =  np.empty((0,0))
for i in dimensions:
    i = in_file.i.astype("unint16")
    np.c_[new_las,i]
print(new_las.shape)
