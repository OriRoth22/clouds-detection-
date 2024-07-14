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
import glob
import csv
from pyproj import Proj
from shapely.geometry import shape        
from shapely.geometry import Polygon
from shapely.ops import orient
from pyproj import Geod
import xy2latlon
import netCDF4 as nc

#%% we wil create image of ch5  ch13 , and create  mixed layer from 13 and 5 channles
#  we will create mask, and then crop it to 512*512 image 


#%%  open files, lat llon , create ch5 and ch13 images remove land
# output: saving png of ch5 and ch13 (seperatly)

def init(file_name,land):
    print ('init')
    
    file =file_name  
    dss=xr.open_dataset(file)
    #add lat  lon values to data det:
    dss = latLon.add_latlon(dss)
    
    # creat images channels 5 and 13
    et = dss.variables['CMI_C05'][:]
    
    
    et = et.where(~np.isnan(land), other=0)  
    
    lat = dss.variables['lat'][:]
    lon = dss.variables['lon'][:]
    plt.imshow(et, extent=[lon.min(), lon.max(), lat.min(),lat.max()])
    fig1 = plt.gcf()
    #plt.show()
    #plt.draw()
    fig1.savefig('ch5.png', dpi=200)
    
    
    et = dss.variables['CMI_C13'][:]
    #land = land.where(~np.isnan(land), other=0)    
    et = et.where(~np.isnan(land), other=0)    
    plt.imshow(et, extent=[lon.min(), lon.max(), lat.min(),lat.max()])
    fig1 = plt.gcf()
    #plt.show()
    #plt.draw()
    fig1.savefig('ch13.png', dpi=200)
    
   


    return dss 

#%% merge the layers: 
#output: png of ch5 and ch13 overlaid 
def merge_layers(savin_name):
    
    print ('merge layers ')

    img1 = cv2.imread("ch5.png")
    img2 = cv2.imread('ch13.png')
    #img3 = cv2.imread('land template.png') # we also using lalnd template to remove clouds above land  
    assert img1 is not None, "file could not be read, check with os.path.exists()"
    assert img2 is not None, "file could not be read, check with os.path.exists()"
    dst = cv2.addWeighted(img1,0.3,img2,0.7,0)
    #dst = cv2.addWeighted(dst,0.8,img3,0.2,0)   
    img = Image.fromarray(dst, "RGB")
    
    # Save the Numpy array as Image
    img.save("merged.png")
    #dst.savefig('merged.png',dpi=200) //  to save in lower resulotion
    img.close()
    return dst
    



#%% convert to binary mask and  // use it if you want the image to be binary image
# get the name of the pocture, return binary image

def convert_to_binary( new_file_name):
    print("convert 2 binary")
    
    
    img = Image.open(new_file_name+".png")
    # Convert the image to grayscale
    img = img.convert('L')

    # Convert the image to binary
    binary_img = img.point(lambda x: 0 if x < 128 else 255, '1')

    # Save the binary image
    #binary_img.save(new_file_name+" binary croped mask .png")   
    return bin



    
    #%% finding contours: 
    # find_contours(file_name, tc_name, file_name+" RGB.png")
    # Find deck borders
def find_contours(file_name, tc_name):
    print("find_contours")
    dss=xr.open_dataset(file_name)
    dss = latLon.add_latlon(dss)

    
    image = cv2.imread("merged.png")  #this is the merged layer of ch5 & ch13
    image2 = cv2.imread(tc_name) #true color image to draw the deck on it 

    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Create the SLIC superpixel object// adjust region_size  for higer resulotion
    #region size is the resulution (e.g 20 means superpixels of  20*20 pixels).
    # ruller = the compactness of the superpixels. larger values will allow more irregular shapes.
    segments = cv2.ximgproc.createSuperpixelSLIC(image, region_size=20, ruler=7)
    
    # Perform the superpixel segmentation
    segments.iterate()
    
    # Get the labels and number of superpixels
    superpixel_labels = segments.getLabels()
    num_superpixels = segments.getNumberOfSuperpixels()
    
    # Create an empty mask for cloud detection
    cloud_mask = np.zeros_like(superpixel_labels, dtype=np.uint8)
    
    
    # Define cloud color criteria in ceLAB space (adjust these values)
    
    # treshholds for the decks detection from the merged image (ch5 and ch13). 
    
    bounderies=3
    if (bounderies==1):
        #a
        lower_cloud_color = ([170, 20, 140])
        upper_cloud_color = ([200, 190, 190])   
    if (bounderies==2): 
        #b
        lower_cloud_color = ([160, 10, 130])
        upper_cloud_color = ([200, 190, 200])
        #c
    if (bounderies==3):
        lower_cloud_color = ([170, 5, 135])
        upper_cloud_color = ([200, 190, 155]) 
    


    #%% Classify superpixels as cloud or non-cloud and creaate mask
    
    for i in range(num_superpixels):
        mask = (superpixel_labels == i)
        mean_lab_color = lab_image[mask].mean(axis=0)
    
        # Check if the mean color of the superpixel falls within the cloud color range
        if (lower_cloud_color <= mean_lab_color).all() and (mean_lab_color <= upper_cloud_color).all():
            cloud_mask[mask] = 255
    
    # Find contours in the cloud mask
    contours, _ = cv2.findContours(cloud_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw cloud contours on the original image
    result_image = image.copy()
    cv2.drawContours(result_image, contours, -1, (0, 0, 255), 2)  # Draw contours in red
    
    #cv2.imwrite(new_file_name+" mereged with contours.png", result_image)
    
    image4= Image.fromarray(result_image, "RGB")
    
    cv2.drawContours(image2, contours, -1, (0, 0, 255), 2)  # Draw contours in red
    
    # Display the image with contours using Matplotlib
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    
    plt.axis('off')  # Turn off axis frame
    
    plt.show()
    
    
    #save the image with contur marked:
    substring = file_name.split('\\')[1] 
    substring = substring.split('.nc')[0]  
    substring= substring+" deck"
    cv2.imwrite(substring+".png", image2)   
    
    # Save the Numpy array as Image
    img = Image.fromarray(result_image, "RGB")
    
    
    #optional - create mask (white background)
    """
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, contours, -1, (255,255,255),thickness=cv2.FILLED)
    """
    
    # handle the items and take the largest contour:

    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True) 
    return sorted_contours ,result_image



#%% measurment of the largest contours 

def measurment(sorted_contours,result_image,index):
    print ("measurment")
    lat = dss.variables['lat'][:]
    lon = dss.variables['lon'][:]
    
    if  0 < len(sorted_contours):
            
        largest_item= sorted_contours[index]

        
        vector = np.vectorize(np.int_)
        
        #calculate size of the largest deck:
            #1. calc the lat/lon of the deck 
            #we need to convert from the x/y of the deck funded to the lat/lon 

        x=largest_item[:,0,0]
        y=largest_item[:,0,1]
        
        fifth_x, fifth_y= x[1],y[1]
        
        
        x=x-150
        x= ((x/930)*1999) # offset of the x asix of the iamge (changes with quality of the iamge)
        x= vector(x)
        
        # Create a mask for elements greater than 2000 
        #beacuse there is maybe some pixels out beacuse of the superpixel algorihm 
        mask = x > 1999
        
        # Replace elements greater than 2000 with 2000
        x[mask] = 1999

        y=y-104 # offset of the y asix of the iamge (changes with quality of the iamge)
        y= ((y/587)*1999)

        y= vector(y)

        mask = y > 1999
        
        y[mask] = 1999
        
        roi_lat=0
        roi_lon=0
        together=[]
        extracted_values=[[],[]]
        
        nodes_x = [151,1080  , 1080 ,151]
        nodes_y = [104 , 104, 690,  690 ]
        nodes_lat =  [   0.196624 ,-45.0904, -41.2323  , 0.190435]
        nodes_lon =     [-110.563 , -69.8311,-139.102, -67.7788]
        
    
            #2. add lat lon to list of cooordinates
        for i in range (len(x)):
            
            roi_lat=(lat[y[i],x[i]].data)
            roi_lon=(lon[y[i],x[i]].data)
            #print ("lat and lon 1" , roi_lat,roi_lon)
            
            #if i==1:
                #lat2,lon2= xy2latlon.interpolate_lat_lon(nodes_x, nodes_y, nodes_lat, nodes_lon, fifth_x, fifth_y)
                #lat3,lon3= xy2latlon.GetLatandLong(0.196624 , -110.563, -45.0904,  -69.8311, 930, 587 ,fifth_y,fifth_x)    
                #print ("lat and lon 2" ,lat2,lon2)
                #print ("lat and lon 3: " ,lat3,lon3)
                
            together.append((roi_lon,roi_lat)) 
            
            if i %10==0:
                extracted_values[0].append (roi_lat.item())
    
                extracted_values[1].append (roi_lon.item())
            
         
            #3. method to calculate size using the list of coordinates
            
        co = {"type": "Polygon", "coordinates": 
              [ together]}
        lonn, latt = zip(*co['coordinates'][0])
        
        pa = Proj("+proj=aea +lat_1=0 +lat_2=-45 +lat_0=-70 +lon_0=-135")
        xx, yy = pa(lonn, latt)
        cop = {"type": "Polygon", "coordinates": [zip(xx, yy)]}
        polygon = Polygon(together)
        geod = Geod(ellps="WGS84")
        poly_area, poly_perimeter = geod.geometry_area_perimeter(orient(polygon))
        
        #print (f"{poly_area:.0f} {poly_perimeter:.0f}")
        cnt = largest_item
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        center_coordinates=(cx,cy)
        """
        # circle the center of the deck
        cv2.circle(result_image, (fifth_x, fifth_y), 10, (0,255,0), 5)
        
        cv2.imshow("ex",result_image) 
        cv2.waitKey(0) #necessary to avoid Python kernel form crashing  
        # closing all open windows 
        cv2.destroyAllWindows() 
        """
        
        # finding the center of the deck in the data      
        cx-=150
        cx=int ((cx/930)*1999)
        cx=min(cx,1999)
        cy-=104
        cy=int ((cy/587)*1999)
        cy=min(cy,1999)
        
        loncenter1,latcenter2 = lon[cy][cx].data, lat[cy][cx].data
        center_coordinates= (loncenter1,latcenter2 )
        #extracted_values =[arr.tolist() for arr in extracted_values]

        return (center_coordinates,poly_area,extracted_values)

    return ((0,0),0,0 ) #in case no deck has been found       


#%%


def create_daily_data(lats, lons, day_index,contour=0):
    # Create coordinates with actual data lengths
    #latnames="latitudes"+ 
    
    lat_coords = xr.DataArray(lats, dims=["latitudes"])
    lon_coords = xr.DataArray(lons, dims=["longitudes"])

    # Create dummy data for the sake of demonstration
    # Replace this with your actual data
    data = np.random.rand(len(lats), len(lons))

    # Create DataArray with filled data and coordinates
    daily_da = xr.DataArray(
        data,
        dims=["latitudes", "longitudes"],
        coords={"latitudes": lat_coords, "longitudes": lon_coords}
    )

    # Create a new Dataset to hold the DataArray and additional coordinates
    daily_ds = xr.Dataset(
        {"data": daily_da},  # Add your data array here
        coords={
            "latitudes": lat_coords,
            "longitudes": lon_coords,
            "time": xr.DataArray([day_index], dims="time")
        }
    )

    return daily_ds


import numpy as np
from netCDF4 import Dataset


def save_to_netcdf(filename, lon1_data, lat1_data, lon2_data, lat2_data, lon3_data, lat3_data, 
                   area1_data, area2_data, area3_data, daynumber_data):
    num_days = len(daynumber_data)
    num_elements = len(lon1_data[0])  # Assuming all lists have the same length
    
    with Dataset(filename, 'w') as nc:
        # Define dimensions
        nc.createDimension('day', num_days)
        nc.createDimension('element', num_elements)

        # Create variables for lon1 and lat1
        for i in range(3):
            var_lon = nc.createVariable(f'lon1_{i}', 'f4', ('day', 'element'))
            var_lon[:, :] = np.array(lon1_data[i])
            var_lat = nc.createVariable(f'lat1_{i}', 'f4', ('day', 'element'))
            var_lat[:, :] = np.array(lat1_data[i])

        # Create variables for lon2 and lat2
        for i in range(3):
            var_lon = nc.createVariable(f'lon2_{i}', 'f4', ('day', 'element'))
            var_lon[:, :] = np.array(lon2_data[i])
            var_lat = nc.createVariable(f'lat2_{i}', 'f4', ('day', 'element'))
            var_lat[:, :] = np.array(lat2_data[i])

        # Create variables for lon3 and lat3
        for i in range(3):
            var_lon = nc.createVariable(f'lon3_{i}', 'f4', ('day', 'element'))
            var_lon[:, :] = np.array(lon3_data[i])
            var_lat = nc.createVariable(f'lat3_{i}', 'f4', ('day', 'element'))
            var_lat[:, :] = np.array(lat3_data[i])

        # Create variables for area1, area2, area3, and daynumber
        var_area1 = nc.createVariable('area1', 'f4', ('day',))
        var_area1[:] = np.array(area1_data)

        var_area2 = nc.createVariable('area2', 'f4', ('day',))
        var_area2[:] = np.array(area2_data)

        var_area3 = nc.createVariable('area3', 'f4', ('day',))
        var_area3[:] = np.array(area3_data)

        var_daynumber = nc.createVariable('daynumber', 'i4', ('day',))
        var_daynumber[:] = np.array(daynumber_data)
    




#%% main function:
    
if __name__ == "__main__":
    
    dsl= xr.open_dataset("000.nc") #SST file, to remove land. 
    land=dsl.variables['SST'][:]   
    
    # paramters: 
    directory_path = "D:/Data/MCMIPF/2020" # change it to wished path
    k=1 #will follow theh day, adjust it to starting day in the year

    # Define the pattern to match the files
    file_pattern = "*.nc"

    # get a list of files that match the pattern
    files = glob.glob(f"{directory_path}/{file_pattern}")
    
    #list will be added as coloumns to the csv file: 
    lon_list=[]
    lat_list=[]
    area=[]
    
    lon_list2=[]
    lat_list2=[]
    area2=[]
    
    lon_list3=[]
    lat_list3=[]
    area3=[]
    
    file_path_list=[]
    epoch_list=[]
    

    values_per_day=[]
    daily_data_arrays = []
    days= np.linspace(364, 366)
    lat1=[]
    lon1=[]
    lat2=[]
    lon2=[]
    lat3=[]
    lon3=[]
    area1=[]
    area2=[]
    area3=[]
    
    
    
    
    filename = "final_data.nc"
    dataset = nc.Dataset(filename, "w", format="NETCDF4")
    
    # Define dimensions (assuming all lists have the same length)
    time_dim = dataset.createDimension("time", None)  # Unlimited length
    list_dim = dataset.createDimension("list_dim", None)
    
    # Create variables for each list
    list1_var = dataset.createVariable("lat1", np.float32, ("time", "list_dim"))
    list2_var = dataset.createVariable("lon1", np.float32, ("time", "list_dim"))
    list3_var = dataset.createVariable("lat2", np.float32, ("time", "list_dim"))
    list4_var = dataset.createVariable("lon2", np.float32, ("time", "list_dim"))
    list5_var = dataset.createVariable("lat3", np.float32, ("time", "list_dim"))
    list6_var = dataset.createVariable("lon3", np.float32, ("time", "list_dim"))    
    # ... Create variables for all 6 lists



    i=1
    start_index=1
    for file_path in files[start_index:366]:
        
        
        #if i==5:
           #break //in case you want to stop in some date
        
        with open(file_path, 'r') as file:
            

            print ("file path = "+ file_path)
            dss= init(file_path,land)
            merge_layers(file_path)
            num=str(k)
            tc_name=num+" True color.png"           
            sorted_contours,result_image= find_contours(file_path, tc_name)
            center_coordinates,poly_area,latlon_together= measurment(sorted_contours,result_image,0)
            poly_area =float(poly_area* pow(10, -6))
            poly_area=abs(poly_area)
            lon_list.append (center_coordinates[0])
            lat_list.append (center_coordinates[1])
            area.append(poly_area)
            
            center_coordinates,poly_area=((0,0),0 )
            if len(sorted_contours)>1:
                center_coordinates,poly_area,latlon_together2= measurment(sorted_contours,result_image,1)
            poly_area =float(poly_area* pow(10, -6))
            poly_area=abs(poly_area)
            lon_list2.append (center_coordinates[0])
            lat_list2.append (center_coordinates[1])
            area2.append(poly_area)
             
            center_coordinates,poly_area=((0,0),0 )
            if len(sorted_contours)>2:
                center_coordinates,poly_area,latlon_together3= measurment(sorted_contours,result_image,2)
            poly_area =float(poly_area* pow(10, -6))
            poly_area=abs(poly_area)
            lon_list3.append (center_coordinates[0])
            lat_list3.append (center_coordinates[1])
            area3.append(poly_area)
            
            
            
            # Creating a row with the data
            substring = file_path.split('0', 1)[0]
            file_path_list.append(substring)
            epoch_list.append(i)
            #add_epoch_data(csv_file_path, i, center_coordinates,poly_area)
        
            #daily_da = create_daily_data(together[0],together[1], i)
            #daily_data_arrays.append(daily_da)


        i=i+1
        k=k+1
        
        #save_to_netcdf("exampls.nc", together[0],together[1],together2[0],together2[1],together3[0],together3[1], area, area2, area3, [364,365])

        lat1.append(latlon_together[0])
        lat2.append(latlon_together2[0])
        lat3.append(latlon_together3[0])
        lon1.append(latlon_together[1])
        lon2.append(latlon_together2[1])
        lon3.append(latlon_together3[1])
        
        list1_var[i, :] = latlon_together[0]
        list2_var[i, :] = latlon_together[1]
        list3_var[i, :] = latlon_together2[0]
        list4_var[i, :] = latlon_together2[1]
        list5_var[i, :] = latlon_together3[0]
        list6_var[i, :] = latlon_together3[1]


      
    dataset.close()  
        #save_to_netcdf("exampls.nc", lat1,lon1,lat2,lon2,lat3,lon3, area, area2, area3, i)
    

        
    # Concatenate DataArrays along the time dimension
    #da = xr.concat(daily_data_arrays, dim="time")

    
    # Save to NC file
    #da.to_netcdf("daily_data1.nc")

    

    # Specify the wished CSV file name
    csv_file_name = "outputs.csv"
    # Combine the lists into a list of tuples
    data = list(zip(epoch_list,file_path_list, lon_list,lat_list, area,together, 
                    lon_list2,lat_list2, area2,together2, lon_list3,lat_list3, area3,together3))
        
    # Open the CSV file in write mode
    with open(csv_file_name, 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)
    
        # Write the header (if needed)
        csv_writer.writerow(["number","Name", "lon1" ," lat1", "area1", "lon2" ," lat2",
                             "area2", "lon3" ," lat3", "area3"])
    
        # Write the data from the lists to the CSV file
        csv_writer.writerows(data)
        
    print(f"Data has been written to {csv_file_name}")
    
    
    
