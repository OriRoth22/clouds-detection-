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

#%% we wil create image of ch5  ch13 , and create  mixed layer from 13 and 5 channles
#  we will create mask, and then crop it to 512*512 image 


#%%  open files, lat llon , create ch5 and ch13 images

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

def merge_layers(savin_name):
    
    print ('merge layers ')

    img1 = cv2.imread("ch5.png")
    img2 = cv2.imread('ch13.png')
    #img3 = cv2.imread('land template.png') # we also using lalnd template to remove clouds above land  
    assert img1 is not None, "file could not be read, check with os.path.exists()"
    assert img2 is not None, "file could not be read, check with os.path.exists()"
    dst = cv2.addWeighted(img1,0.4,img2,0.6,0)
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
    
def find_contours(file_name, tc_name):
    print("find_contours")
    dss=xr.open_dataset(file_name)
    dss = latLon.add_latlon(dss)

    
    image = cv2.imread("merged.png")  #this is the merged layer of ch5 & ch13
    image2 = cv2.imread(tc_name) #true color image to draw the deck on it 

    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Create the SLIC superpixel object// adjust region_size  for higer resulotion
    segments = cv2.ximgproc.createSuperpixelSLIC(image, region_size=20, ruler=4.0)
    
    # Perform the superpixel segmentation
    segments.iterate()
    
    # Get the labels and number of superpixels
    superpixel_labels = segments.getLabels()
    num_superpixels = segments.getNumberOfSuperpixels()
    
    # Create an empty mask for cloud detection
    cloud_mask = np.zeros_like(superpixel_labels, dtype=np.uint8)
    
    
    # Define cloud color criteria in LAB space (adjust these values)
    
    #1:
    lower_cloud_color = np.array([80, 100, 100])
    upper_cloud_color = np.array([170, 200, 200])  
      
    """
    #2:
    lower_cloud_color = ([90, -10, 130])
    upper_cloud_color = ([180, 150, 190])

    """
   
    """
    #3:
    lower_cloud_color = np.array([120, -50, 120])
    upper_cloud_color = np.array([150, 120, 160])
    """
    
    lower_cloud_color = ([160, 10, 130])
    upper_cloud_color = ([200, 190, 200])

    lower_cloud_color = ([150, 0, 130])
    upper_cloud_color = ([210, 200, 200])    
    
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
    substring = file_name.split('/')[3] 
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

def measurment(sorted_contours,result_image):
    print ("measurment")
    lat = dss.variables['lat'][:]
    lon = dss.variables['lon'][:]
    
    if  0 < len(sorted_contours):
            
        largest_item= sorted_contours[0]
        largest_item2=sorted_contours[1]
        largest_item3=sorted_contours[2]
        
        vector = np.vectorize(np.int_)
        
        #calculate size of the largest deck:
            #1. calc the lat/lon of the deck 
            #we need to convert from the x/y of the deck funded to the lat/lon 

        x=largest_item[:,0,0]
        x=x-165
        x= ((x/930)*2000) # offset of the x asix of the iamge (changes with quality of the iamge)
        x= vector(x)
        
        y=largest_item[:,0,1]

        y=y-105  # offset of the y asix of the iamge (changes with quality of the iamge)
        y= ((y/604)*2000)
        y= vector(y)

        roi_lat=0
        roi_lon=0
        together=[]
            
            #2. add lat lon to list of cooordinates
        for i in range (len(x)):
            
            roi_lat=(lat[y[i],x[i]].data)
            roi_lon=(lon[y[i],x[i]].data)
            together.append((roi_lon,roi_lat))         
         
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
        
        
        
        #%%finding the center of the deck
        cnt = largest_item
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        center_coordinates=(cx,cy)
        # circle the center of the deck
        cv2.circle(result_image, center_coordinates, 10, (0,255,0), 5)
        
        # finding the center of the deck in the data      
        cx-=157
        cx=int ((cx/930)*2000)
        cy-=100
        cy=int ((cy/604)*2000)
        
        loncenter1,latcenter2 = lon[cy][cx].data, lat[cy][cx].data
        center_coordinates= (loncenter1,latcenter2 )
        return (center_coordinates,poly_area)

    return ((0,0),0 ) #in case no deck has been found
                

#%% main function:
    
if __name__ == "__main__":
    
    dsl= xr.open_dataset("000.nc")
    land=dsl.variables['SST'][:]   
    
    # paramters: 
    directory_path = "D:/Data/MCMIPF/2021" # change it to wished path
    k=1 #will follow theh day, adjust it to starting day in the year

    # Define the pattern to match the files
    file_pattern = "*.nc"

    # get a list of files that match the pattern
    files = glob.glob(f"{directory_path}/{file_pattern}")
    
    #list will be added as coloumns to the csv file: 
    lon_list=[]
    lat_list=[]
    area=[]
    file_path_list=[]
    epoch_list=[]
    i=1
    start_index=1

    
    for file_path in files[start_index:]:
        
        #if i==5:
           #break //in case you want to stop in some date
        
        with open(file_path, 'r') as file:
            

            print ("file path = "+ file_path)
            dss= init(file_path,land)
            merge_layers(file_path)
            num=str(k)
            tc_name=num+" True color.png"           
            sorted_contours,result_image= find_contours(file_path, tc_name)
            center_coordinates,poly_area= measurment(sorted_contours,result_image)
            poly_area =float(poly_area* pow(10, -6))
            lon_list.append (center_coordinates[0])
            lat_list.append (center_coordinates[1])

            area.append(poly_area)
            # Creating a row with the data
            substring = file_path.split('0', 1)[0]
            file_path_list.append(substring)
            epoch_list.append(i)
            #add_epoch_data(csv_file_path, i, center_coordinates,poly_area)
        i=i+1
        k=k+1

    
    # Combine the lists into a list of tuples
    data = list(zip(epoch_list,file_path_list, lon_list,lat_list, area))
    
    # Specify the wished CSV file name
    csv_file_name = "output 2020 Fin.csv"
    
    # Open the CSV file in write mode
    with open(csv_file_name, 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)
    
        # Write the header (if needed)
        csv_writer.writerow(["number","Name", "lon" ," lat", "area"])
    
        # Write the data from the lists to the CSV file
        csv_writer.writerows(data)
    
    print(f"Data has been written to {csv_file_name}")

            
