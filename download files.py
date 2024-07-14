# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:57:28 2024

@author: TOSHIBA
"""
import xarray as xr 
#import s3fs
import numpy as np
import scipy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as plt
from netCDF4 import Dataset
import numpy as np
import netCDF4 
#import gini
import xarray as xr
import warnings
from PIL import Image
import cv2
warnings.filterwarnings('ignore')
import s3fs
from datetime import datetime
import os



def download (name_of_file):
    print ('downlaod')
    myobj = datetime.now()
    print("Current hour ", myobj)
    # Use the anonymous credentials to access public data
    fs = s3fs.S3FileSystem(anon=True)
    # List contents of GOES-16 bucket.
    fs.ls('s3://noaa-goes16/')
    # List specific files of GOES-16 CONUS data (multiband format) on a certain hour
    files = np.array(fs.ls(name_of_file))
    
    # Download the first file, and rename it the same name (without the directory structure)
    fs.get(files[0], files[0].split('/')[-1])
    
    
    name=files[0].split('/')[-1]
    return name

# you need to decide which product you want to download and what day/s and hour
# You can download manually from https://noaa-goes16.s3.amazonaws.com/index.html
# for products list read here: https://docs.opendata.aws/noaa-goes16/cics-readme.html
# we used MCMIPF for creatng clouds, ACMF for cloud cover , SSTF for land removal
# the files will be sved - the whole area and file sliced the the desired area
# the code will be saved in the current folder

base_string = 'noaa-goes16/ABI-L2-MCMIPF/2020/{}/17/' #adjust it to desierd year and hour

for i in range(366, 367): #range of wished days to downlaod adjust it to desired days
    # Format the number to havMCMIPFe three digits with leading zeros
    number_str = str(i).zfill(3)
    # Replace the placeholder "{}" with the formatted number
    new_string = base_string.format(number_str)
    print(new_string)
    file = download(new_string)
    ds=xr.open_dataset(file)
    subset = ds.sel(x=slice(-0.1,0.0161), y=slice(0.00063,-0.11136))
    print ('saving to ', str(i).zfill(3)+'.nc')
    subset.to_netcdf(path=str(i).zfill(3)+'.nc')
    ds.close()
    os.remove(file)  # if you want to delete the original file


