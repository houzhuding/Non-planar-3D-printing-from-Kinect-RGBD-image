""" STL file processsing """
import numpy as np  
from stl import mesh
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
# from toolpathGenerationFromContour import *


import numpy as np
import math
from shapely.geometry import Polygon, Point
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class toolpathGenerationFromContour3D:
	def __init__(self,cnt,mesh,step,feed):
		self.mesh = mesh
		self.cnt = cnt
		self.step = step
		self.feed = feed
		self.contour = cnt
		self.xPoints = cnt[:,0]
		self.yPoints = cnt[:,1]

	def parseSTL(self,STL):
		veclen = len(self.mesh.vectors)
		# for i in range(veclen):

	def scanHorizental(self):
		
		poly  = Polygon(self.cnt)
		edge_pts = []
		## define the horizontal toolpath
		# xGrid = np.arange(0,int(1.2*max(self.xPoints)),self.step)
		# yGrid = np.arange(0,int(1.2*max(self.yPoints)),self.feed)

		xmin = min(self.xPoints)
		xmax = max(self.xPoints)
		ymin = min(self.yPoints)
		ymax = max(self.yPoints)

		xGrid = np.linspace(xmin, xmax,(xmax-xmin)/self.step)
		yGrid = np.linspace(ymin, ymax,(ymax-ymin)/self.step)

		print(len(xGrid))
		X, Y = np.meshgrid(xGrid, yGrid)
		veclen = len(self.mesh.vectors)

		vectors_valid = [self.mesh.vectors[i] for i in range(veclen) 
		if min(self.mesh.vectors[i][0][0],self.mesh.vectors[i][1][0],self.mesh.vectors[i][2][0])>xmin
		and max(self.mesh.vectors[i][0][0],self.mesh.vectors[i][1][0],self.mesh.vectors[i][2][0])<xmax
		and min(self.mesh.vectors[i][0][1],self.mesh.vectors[i][1][1],self.mesh.vectors[i][2][1])>ymin
		and max(self.mesh.vectors[i][0][1],self.mesh.vectors[i][1][1],self.mesh.vectors[i][2][1])<ymax]

		points_3d = np.array(vectors_valid)

		# print(points_3d,points_3d.shape)

		xs = points_3d[:,:,0].flatten()
		ys = points_3d[:,:,1].flatten()
		zs = points_3d[:,:,2].flatten()

		# print(xs,xs.shape)
		Z = griddata((xs, ys), zs, (X, Y),method='linear')


		for j in range(len(yGrid)):
			for i in range(len(xGrid)):
				
				if j%2==1:
					idx = len(xGrid)-1-i  
				else:
					idx = i
				if idx < len(xGrid)-1:
					pt1 = Point(xGrid[idx],yGrid[j])
					pt2 = Point(xGrid[idx+1],yGrid[j])
					ptz = Z[j][idx]
					## from left to right
					if ~np.isnan(ptz):
						if pt1.within(poly):
							edge_pts.append((pt1.x,pt1.y,ptz))
						if pt1.within(poly)==False and pt2.within(poly)==True:
							edge_pts.append(((pt1.x+pt2.x)/2.0,pt1.y,ptz))
							
						if pt1.within(poly)==True and pt2.within(poly)==False:
							edge_pts.append(((pt1.x+pt2.x)/2.0,pt1.y,ptz))
		# edge_pts.extend(self.contour)

		for j in range(len(xGrid)):
			for i in range(len(yGrid)):
				
				if j%2==1:
					idx = len(yGrid)-1-i  
				else:
					idx = i
				if idx < len(yGrid)-1:
					pt1 = Point(xGrid[j],yGrid[idx])
					pt2 = Point(xGrid[j],yGrid[idx+1])
					ptz = Z[idx][j]
					## from left to right
					if ~np.isnan(ptz):
						if pt1.within(poly):
							edge_pts.append((pt1.x,pt1.y,ptz))
						if pt1.within(poly)==False and pt2.within(poly)==True:
							edge_pts.append(((pt1.x+pt2.x)/2.0,pt1.y,ptz))
							
						if pt1.within(poly)==True and pt2.within(poly)==False:
							edge_pts.append(((pt1.x+pt2.x)/2.0,pt1.y,ptz))
		return np.array(edge_pts),X,Y,Z

	def contour2path(self,path,fileName,):
		scale_factor = 0.16;
		offset_x = 0
		offset_y = 0
		offset_z = 0.5
		xyz = path
		coordsXYZ = xyz*scale_factor
		# print coordsXYZ,coordsXYZ.shape
		coordsXYZ[:,0] = coordsXYZ[:,0]+ offset_x# move the whole strucuture along X direction by a offset
		## Compute the path length and extrusion angle between each adjacent points
		## initialize parameters
		default_speed = 100; # mm/min
		user_input_angular_vel = 0.1; # mL/hr
		Angle = []
		Angle.append(0)
		for i in range(1,len(coordsXYZ)):
		    cur_path = np.linalg.norm(coordsXYZ[i-1,:]-coordsXYZ[i,:])
		    speed = default_speed
		    time = cur_path/speed
		    angle = time*user_input_angular_vel
		    if  i< len(coordsXYZ)-1:
		        Angle.append(Angle[i-1]-angle)
		    else:
		        Angle.append(Angle[i-1])
		## Convert the toolpath to G-code
		## open the file
		fileIDx=open(fileName,'w');  #file where comands are appended
		# value = ('X',coordsXYZ[0,0],coordsXYZ[1,0],coordsXYZ[2,0],Angle[0],default_speed)
		formatSpec= 'X%7.6f Y%7.6f Z%7.6f A%7.6f F%7.6f\n';
		## initial the G-code file
		fileIDx.write('\n')
		fileIDx.write('G90\nG49\n')
		fileIDx.write('G1 X0 Y0 Z0 F60\n')#mm/min
		## fprintf(fileIDx,'G1 X0 Y0 Z3')
		## add transion path in front and end 
		fileIDx.write(formatSpec %(0, 0 ,offset_z ,Angle[0] ,default_speed,))
		fileIDx.write(formatSpec %(coordsXYZ[0,0], coordsXYZ[0,1] ,offset_z ,Angle[0] ,default_speed,))
		fileIDx.write(formatSpec %(coordsXYZ[0,0], coordsXYZ[0,1] ,coordsXYZ[0,2] ,Angle[0] ,default_speed,))
		### write down the coordinates one by one in format: G1 X Y Z A F\
		for i in range(len(coordsXYZ)):
			fileIDx.write(formatSpec %(coordsXYZ[i,0], coordsXYZ[i,1] ,coordsXYZ[i,2] ,Angle[i] ,default_speed,))
		
		fileIDx.write(formatSpec %(coordsXYZ[i,0], coordsXYZ[i,1] ,offset_z ,Angle[i] ,default_speed,))
		fileIDx.write(formatSpec %(0, 0 ,offset_z ,Angle[i] ,default_speed,))
		fileIDx.write('G1 X0 Y0 Z0 F100\n')#mm/min
		fileIDx.write('M05\nM02\n');
		fileIDx.close()
