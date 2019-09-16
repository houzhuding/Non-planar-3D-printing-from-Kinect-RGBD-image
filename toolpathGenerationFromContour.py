"""
Generate tool path from the extracted contour by using the continuous tool-path 
generator

Tool-path 
"""

import numpy as np
import math
from shapely.geometry import Polygon, Point
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
class toolpathGenerationFromContour:
	def __init__(self,cnt, step,feed):
		self.cnt = cnt
		self.step = step
		self.feed = feed
		self.contour = cnt
		self.xPoints = cnt[:,0]
		self.yPoints = cnt[:,1]
	def scanHorizental(self):
		
		poly  = Polygon(self.cnt)
		edge_pts = []
		## define the horizontal toolpath
		xGrid = np.arange(0,int(1.2*max(self.xPoints)),self.step)
		yGrid = np.arange(0,int(1.2*max(self.yPoints)),self.feed)
		for j in range(len(yGrid)):
			for i in range(len(xGrid)):
				if j%2==1:
					idx = len(xGrid)-1-i  
				else:
					idx = i
				if idx < len(xGrid)-1:
					pt1 = Point(xGrid[idx],yGrid[j])
					pt2 = Point(xGrid[idx+1],yGrid[j])

					## from left to right
					if pt1.within(poly)==False and pt2.within(poly)==True:
						edge_pts.append(((pt1.x+pt2.x)/2.0,pt1.y))
						
					if pt1.within(poly)==True and pt2.within(poly)==False:
						edge_pts.append(((pt1.x+pt2.x)/2.0,pt1.y))
		## define the vertical toolpath
		edge_pts.extend(self.contour)
		# print (edge_pts)
		xGrid = np.arange(0,int(1.2*max(self.xPoints)),self.feed)
		yGrid = np.arange(0,int(1.2*max(self.yPoints)),self.step)
		for j in range(len(xGrid)):
			for i in range(len(yGrid)):
				if j%2==1:
					idx = len(yGrid)-1-i  
				else:
					idx = i
				if idx < len(yGrid)-1:
					pt1 = Point(xGrid[j],yGrid[idx])
					pt2 = Point(xGrid[j],yGrid[idx+1])

					## from left to right
					if pt1.within(poly)==False and pt2.within(poly)==True:
						edge_pts.append(((pt1.x+pt2.x)/2.0,pt1.y))
						
					if pt1.within(poly)==True and pt2.within(poly)==False:
						edge_pts.append(((pt1.x+pt2.x)/2.0,pt1.y))
				 				
		 						# print edge_pts
		return np.array(edge_pts)
	def get_z_path(self,box):
		"""
		get z path according to longside
		"""
		### the rectangle has 4 points
		### rect[0][0] rect[0][1]
		### 
		### rect[1][0] rect[1][1]
		side1 = np.linalg.norm(box[0]-box[1])
		side2 = np.linalg.norm(box[0]-box[3])
		xp1 = box[0]
		if side1 > side2:
			xp2 = box[3]
		else:
			xp2 = box[1]


		longside = max(side1,side2)
		shortside = min(side1,side2)
		z = len(self.cnt)*[0]
		print(side1,side2,box)
		### convert to the rectangle coordinates
		if self.cnt is not None:
			cnt_in_box = self.cnt

			x = cnt_in_box[:,1]
			y = cnt_in_box[:,0]
			y0 = cnt_in_box[0,0]
			mid = (max(y)-min(y))/2
			ygap = max(y)-min(y)
			ymin = min(y)

			z = [(y[i]-ymin)/2 if (np.cross(xp2-xp1, cnt_in_box[i,:]-xp1))/np.linalg.norm(xp2-xp1)<longside/2 
					else (ygap-y[i]+ymin)/2 for i in range(len(y))]
			### fit the long side
		return z 
		### the relationship z =f(x,y)
		# R = 43 # mm
		# d = 33 # mm
		# h = 15 # mm




###  calibration of the coordinate by timing a scale factor
	def contour2path(self,path,fileName,):
		## Calibration of the coordinates by timing a scale factor
		scale_factor = 0.16; #scale_factor is the Real_dimension/Set_dimension
		offset_x       = 0.0; # offset of the scaffold
		offset_y       = 0.0;
		offset_z       = 0.5; 

		# starting points p1 p2

		coordsXYZ = path*scale_factor
		coordsXYZ[:,0] = coordsXYZ[:,0]+ offset_x# move the whole strucuture along X direction by a offset
		startPts = [[0,0,0],[0,0,offset_z],[coordsXYZ[0,0], coordsXYZ[0,1] ,offset_z]]
		endPts = [[coordsXYZ[0,0], coordsXYZ[0,1] ,offset_z],[0,0,offset_z],[0,0,0]]
		coordsXYZ = np.concatenate((startPts,coordsXYZ,endPts),axis=0)

		#  Compute the path length and extrusion angle between each adjacent points
		#  initialize parameters
		default_speed_set = 10; # mm/s 
		speed_factor = 10;
		default_speed = default_speed_set*speed_factor;
		# speed_inc = 10;
		# compute the flowrate 
		user_input_angular_vel = 1; # mL/hr
		Angle = []
		Times = [0]
		Angle.append(0)
		for i in range(1,len(coordsXYZ)):
		    cur_path = np.linalg.norm(coordsXYZ[i-1,:]-coordsXYZ[i,:])
		    speed = default_speed
		    time = cur_path/speed
		    Times.append(time*60)
		    angle = time*user_input_angular_vel
		    if  i< len(coordsXYZ)-1:
		        Angle.append(Angle[i-1]-angle)
		    else:
		        Angle.append(Angle[i-1])

		## Convert the toolpath to G-code
		# plot the tool-path in python
		## open the file
		fileIDx=open(fileName,'w');  #file where comands are appended
		# value = ('X',coordsXYZ[0,0],coordsXYZ[1,0],coordsXYZ[2,0],Angle[0],default_speed)
		formatSpec= 'X%7.6f Y%7.6f Z%7.6f A%7.6f F%7.6f;%7.6f\n';
		## initial the G-code file
		fileIDx.write('G90\nG49\n')
		fileIDx.write('G1 X0 Y0 Z0 F60\n')#mm/min
		## fprintf(fileIDx,'G1 X0 Y0 Z3')
		## add transion path in front and end 
		# fileIDx.write(formatSpec %(0, 0 ,offset_z ,Angle[0] ,default_speed,0))
		# fileIDx.write(formatSpec %(coordsXYZ[0,0], coordsXYZ[0,1] ,offset_z ,Angle[0] ,default_speed,0))
		# fileIDx.write(formatSpec %(coordsXYZ[0,0], coordsXYZ[0,1] ,0 ,Angle[0] ,default_speed,0))
		### write down the coordinates one by one in format: G1 X Y Z A F\
		for i in range(len(coordsXYZ)):
		    fileIDx.write(formatSpec %(coordsXYZ[i,0], coordsXYZ[i,1] ,coordsXYZ[i,2] ,Angle[i] ,default_speed,Times[i]))

		fileIDx.write(formatSpec %(coordsXYZ[i,0], coordsXYZ[i,1] ,offset_z ,Angle[i] ,default_speed,Times[i]))
		fileIDx.write(formatSpec %(0, 0 ,offset_z ,Angle[i] ,default_speed,Times[i]))
		fileIDx.write('G1 X0 Y0 Z0 F100\n')#mm/min
		fileIDx.write('M05\nM02\n');
		fileIDx.close()
		

# coords = np.array([(1, 1), (10, 1), (5, 10)])
# # poly = Polygon(coords)
# feed = 1
# step = 0.1
# toolPath = toolpathGenerationFromContour(coords,step,feed)
# path =  toolPath.scanHorizental()
# pathFile = 'test.nc'
# pathFile = toolPath.contour2path(path,pathFile)
# figure = Figure(figsize=(3,3), dpi=100) 
# a = figure.add_subplot(111)
# plt.plot(path[:,0],path[:,1])
# plt.show()