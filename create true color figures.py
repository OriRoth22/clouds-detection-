# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:52:06 2024

@author: USER
"""
import glob

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from PIL import Image
import cv2
import netCDF4
import pandas as pd
import numpy as np
import scipy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import latLon
# Specify the directory where your files are located
directory_path = "D:/Data/MCMIPF/2020"

# Define the pattern to match the files
file_pattern = "*.nc"

# Use glob to get a list of files that match the pattern
files = glob.glob(f"{directory_path}/{file_pattern}")
k=1
# Iterate over the files and perform your desired actions
for file_path in files[1:]:
    # Your code to process each file goes here
    # For example, you can open the file and perform some operations
    
    with open(file_path, 'r') as file:
           
            print (file_path )

            dss=xr.open_dataset(file_path)
            new_file_name=file_path
            savin_name=new_file_name+".png"
    
            #add lat  lon values to data det:
            dss = latLon.add_latlon(dss)
    
            lat = dss.variables['lat'][:]
            lon = dss.variables['lon'][:]
    
    
            #%%      Create true color
            r = dss['CMI_C02'].data; r = np.clip(r, 0, 1)
            g = dss['CMI_C03'].data; g = np.clip(g, 0, 1)
            b = dss['CMI_C01'].data; b = np.clip(b, 0, 1)
    
            # Apply a gamma of 2.5
            gamma = 2.5; r = np.power(r, 1/gamma); g = np.power(g, 1/gamma); b = np.power(b, 1/gamma)
            # Calculate "true green" from the veggie channel
            g_true = 0.45 * r + 0.1 * g + 0.45 * b
            g_true = np.clip(g_true, 0, 1)
            rgb = np.dstack((r, g_true, b))
    
    
            p2=plt.imshow(rgb,extent=[lon.min(), lon.max(), lat.min(),lat.max()])
    
    
            fig1 = plt.gcf()
            #fig1.set_size_inches(20, 21)
            plt.show()
    
            plt.draw()
            #substring = new_file_name.split('ng', 1)[1]
            k=str(k)
            fig1.savefig(k+" True color.png",dpi=200)
            k=int (k)
            k+=1
            


        