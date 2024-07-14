# -*- coding: utf-8 -*-
"""
this code is to convert a place on the map to lat lon, 
first you need to feel the edges of the image, in our project 

coordinations:
North Wast: 0.196624 , -110.563
North West: 0.190435 , -69.8311
South East : -45.0904 , -139.102
South West: -41.2323, -67.7788

"""
from scipy.interpolate import griddata

def interpolate_lat_lon(nodes_x, nodes_y, nodes_lat, nodes_lon, fifth_x, fifth_y):
    # Creating the input data for interpolation
    points = list(zip(nodes_x, nodes_y))
    values_lat = nodes_lat
    values_lon = nodes_lon

    # Interpolate the latitude and longitude of the fifth node
    fifth_lat = griddata(points, values_lat, (fifth_x, fifth_y), method='cubic')
    fifth_lon = griddata(points, values_lon, (fifth_x, fifth_y), method='cubic')

    return fifth_lat, fifth_lon


def GetLatandLong(top_Left_Lat, top_Left_Long, bottom_Right_Lat, bottom_Right_Long,img_Width,img_Height, target_Top, target_Left):
    diff_Between_Top_Bottom_Lat = bottom_Right_Lat - top_Left_Lat
    percentage_Of_Total_Lat_In_Picture = diff_Between_Top_Bottom_Lat/90*100
    image_Size_Height_Required_To_Cover_Entire_Earth = img_Height/percentage_Of_Total_Lat_In_Picture*100
    top_Left_Percentage_Of_Lat = top_Left_Lat/90*100
    top_Left_Pixel_In_Image = image_Size_Height_Required_To_Cover_Entire_Earth*top_Left_Percentage_Of_Lat/100
    target_Pixel_In_Whole_Earth_Image = top_Left_Pixel_In_Image + target_Top
    percentage_Of_Target_In_Image = target_Pixel_In_Whole_Earth_Image/image_Size_Height_Required_To_Cover_Entire_Earth*100    
    target_Lat = percentage_Of_Target_In_Image*90/100


    diff_Between_Top_Bottom_Long = bottom_Right_Long - top_Left_Long
    percentage_Of_Total_Long_In_Picture = diff_Between_Top_Bottom_Long/180*100
    image_Size_Width_Required_To_Cover_Entire_Earth = img_Width/percentage_Of_Total_Long_In_Picture*100
    top_Left_Percentage_Of_Long = top_Left_Long/180*100
    top_Left_Pixel_In_Image = image_Size_Width_Required_To_Cover_Entire_Earth*top_Left_Percentage_Of_Long/100
    target_Pixel_In_Whole_Earth_Image = top_Left_Pixel_In_Image + target_Left
    percentage_Of_Target_In_Image = target_Pixel_In_Whole_Earth_Image/image_Size_Width_Required_To_Cover_Entire_Earth*100    
    target_Long = percentage_Of_Target_In_Image*180/100






    return target_Lat,target_Long

if __name__ == "__main__":
    nodes_x = [151,1080  , 1080 ,151]
    nodes_y = [104, 690 , 104,  690 ]
    nodes_lat =  [   0.196624,   -41.2323    ,-45.0904 , 0.190435 ]
    nodes_lon =     [-110.563   , -67.7788   ,-139.102, -69.8311]
    
    fifth_x = 1311  # X-coordinate of the fifth node
    fifth_y = 2196  # Y-coordinate of the fifth node
    
    fifth_lat, fifth_lon = interpolate_lat_lon(nodes_x, nodes_y, nodes_lat, nodes_lon, fifth_x, fifth_y)
    
    print(f"Interpolated Latitude of the node: {fifth_lat}")
    print(f"Interpolated Longitude of the node: {fifth_lon}")

