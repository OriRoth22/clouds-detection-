Files:
Python:
a.	Download files – 
b.	Create true color 
c.	Find and measure contours
d.	Calc sky map cells area

Other helpful files:
e.	Slice – case you download manually nc file and want to slice it to the area.
f.	Latlon – this function add latlon values to the nc file of goes
g.	Xy to lat lon – this function get xy and return the lat lon 
Nc files:
	MCMIPF for creating decks and true color map,
 ACMF for cloud cover ,
 SSTF for land removal

A.	Download and slice the files:
2.	Download one/multiply files from aws. 

file number1 - download.
For instructions and products summary - https://docs.opendata.aws/noaa-goes16/cics-readme.html

3.	Slice the nc files to the wanted area.

This is also be done in the file namaed " Download files"
	subset = ds.sel (x=slice(-0.1,0.0161), y=slice(0.00063,-0.11136))

B.	Create true color and decks :
4.	 All in file " Find and measure contours" 
Add lat / lon data to the files
 based on https://lsterzinger.medium.com/add-lat-lon-coordinates-to-goes-16-goes-17-l2-data-and-plot-with-cartopy-27f07879157f

5.	Create true color images from the nc files.

seperate code file takes the  1th, 2th, 3th  channels  and combine them to crate true color image

6.	Take 5th channel and 13th channel and merge them.
Remove the land with file of sst named "000.nc"
Merge layer's function.
7.	Use super pixel with threshold to find SC decs. 
8.	Masure the area of the big Deck.
Measurement function
9.	Option - Create mask of all the decks for each day.


10.	Option – DL algorithm to find deck – based on the data of the super pixel algorithm. 
In unet folder

C.	Measure the cloudy area:

11.	 count the cloud cover for each day, compared to the big deck. Use the file named "Calc sky map cells area" you need also a file of "SST" to remove the land, use "000.nc" file.





26°36'26"S 127°54'08"W


