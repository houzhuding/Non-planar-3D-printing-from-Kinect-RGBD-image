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

	def map_points2trianle(self,vectors_valid,xGrid,yGrid):
		"""
		loop each triangles and find corresponding points in it

		"""
		# class dPoint():
		# 	def __init__(self,x,y,tri):
		# 		self._x = x
		# 		self._y = y
		# 		self.pt = (x,y)
		# 		self.parent = tri

		# pair = dPoint(x,y,tri)	

		point_tri_dict = {}
		for triangle in vectors_valid:
		### form a plane using 3 points of a triangle

			p1 = triangle[0,:]
			p2 = triangle[1,:]
			p3 = triangle[2,:]

			x1 = p1[0]
			y1 = p1[1]
			x2 = p2[0]
			y2 = p2[1]
			x3 = p3[0]
			y3 = p3[1]

			area = abs((x1*y2+x2*y3+x3*y1-x1*y3-x2*y1-x3*y2))
			tri = np.array([p1[:2],p2[:2],p3[:2],p1[:2]])
			# print(tri)
			poly = Polygon(tri)
			print(area)
			if area > 1e-4:

				xmin = min(triangle[:,0])
				xmax = max(triangle[:,0])
				ymin = min(triangle[:,1])
				ymax = max(triangle[:,1])
				x_sub_Grid = [x for x in xGrid if x>xmin and x<=xmax]
				y_sub_Grid = [y for y in yGrid if y>ymin and y<=ymax]


				for i in range(len(x_sub_Grid)):
					for j in range(len(y_sub_Grid)):
						pt = Point(x_sub_Grid[i],y_sub_Grid[j])
						if pt.within(poly) and (pt.x,pt.y) not in point_tri_dict:# the point is in this triangle
							point_tri_dict[(pt.x,pt.y)] = triangle

		return point_tri_dict



	def calculate_z (self,x,y,vectors_valid):
		"""
		find valid intersection point xy and vectors_valid

		 det[x y z 1; 
		     x_1 y_1 z_1 1;   
		     x_2 y_2 z_2 1; 
		     x_3 y_3 z_3 1]=0

		(1) x	=	x_4+(x_5-x_4)t	
		(2) y	=	y_4+(y_5-y_4)t	
		(3) z	=	z_4+(z_5-z_4)t

		where :

		 t=-(|1 1 1 1; 
		 x_1 x_2 x_3 x_4; 
		 y_1 y_2 y_3 y_4; 
		 z_1 z_2 z_3 z_4|)
		 /
		 (|1 1 1 0; 
		 x_1 x_2 x_3 x_5-x_4; 
		 y_1 y_2 y_3 y_5-y_4; 
		 z_1 z_2 z_3 z_5-z_4|). 

		"""
		### loop for each facet
		pzs = []
		for triangle in vectors_valid:
			### detect if the xy is located in the projection of certain triangle (or line)

			### form a plane using 3 points of a triangle
			p1 = triangle[0,:]
			p2 = triangle[1,:]
			p3 = triangle[2,:]
			tri = np.array([p1[:2],p2[:2],p3[:2],p1[:2]])
			# print(tri)
			poly = Polygon(tri)
			if Point(x,y).within(poly):
				# compute the normal
				#normal = np.cross(p1-p2,p1-p3)
				
				# define two points of vertical line
				p4 = [x,y,0]
				p5 = [x,y,1]

				## solve t 
				upper = np.array([[1,1,1,1],
								  [p1[0],p2[0],p3[0],p4[0]],
								  [p1[1],p2[1],p3[1],p4[1]],
								  [p1[2],p2[2],p3[2],p4[2]],
								  ])
				lower = np.array([[1,1,1,0],
								  [p1[0],p2[0],p3[0],p5[0]-p4[0]],
								  [p1[1],p2[1],p3[1],p5[1]-p4[1]],
								  [p1[2],p2[2],p3[2],p5[2]-p4[2]],
								  ])
				t = -np.linalg.det(upper)/np.linalg.det(lower)
				pz = p4[2]+(p5[2]-p4[2])*t
				pzs.append(pz)
				trianles

		if pzs is not None:
			return np.mean(pzs)
		else:
			return float('nan')


	def calculate_z_v2 (self,point,triangle):
		"""
		find valid intersection point xy and vectors_valid

		 det[x y z 1; 
		     x_1 y_1 z_1 1;   
		     x_2 y_2 z_2 1; 
		     x_3 y_3 z_3 1]=0

		(1) x	=	x_4+(x_5-x_4)t	
		(2) y	=	y_4+(y_5-y_4)t	
		(3) z	=	z_4+(z_5-z_4)t

		where :

		 t=-(|1 1 1 1; 
		 x_1 x_2 x_3 x_4; 
		 y_1 y_2 y_3 y_4; 
		 z_1 z_2 z_3 z_4|)
		 /
		 (|1 1 1 0; 
		 x_1 x_2 x_3 x_5-x_4; 
		 y_1 y_2 y_3 y_5-y_4; 
		 z_1 z_2 z_3 z_5-z_4|). 

		"""
		### loop for each facet
		# pz = float('nan')
		### detect if the xy is located in the projection of certain triangle (or line)

		### form a plane using 3 points of a triangle
		p1 = triangle[0,:]
		p2 = triangle[1,:]
		p3 = triangle[2,:]
		# tri = np.array([p1[:2],p2[:2],p3[:2],p1[:2]])
		# print(tri)
		# poly = Polygon(tri)
		# if Point(x,y).within(poly):
			# compute the normal
			#normal = np.cross(p1-p2,p1-p3)
			
			# define two points of vertical line
		p4 = [point.x,point.y,0]
		p5 = [point.x,point.y,-1]

		## solve t 
		upper = np.array([[1,1,1,1],
						  [p1[0],p2[0],p3[0],p4[0]],
						  [p1[1],p2[1],p3[1],p4[1]],
						  [p1[2],p2[2],p3[2],p4[2]],
						  ])
		lower = np.array([[1,1,1,0],
						  [p1[0],p2[0],p3[0],p5[0]-p4[0]],
						  [p1[1],p2[1],p3[1],p5[1]-p4[1]],
						  [p1[2],p2[2],p3[2],p5[2]-p4[2]],
						  ])
		t = -np.linalg.det(upper)/np.linalg.det(lower)
		pz = p4[2]+(p5[2]-p4[2])*t

		# if pz==0:
		# print(point,pz)
		return pz


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
		if min(self.mesh.vectors[i][0][0],self.mesh.vectors[i][1][0],self.mesh.vectors[i][2][0])>=xmin
		and max(self.mesh.vectors[i][0][0],self.mesh.vectors[i][1][0],self.mesh.vectors[i][2][0])<=xmax
		and min(self.mesh.vectors[i][0][1],self.mesh.vectors[i][1][1],self.mesh.vectors[i][2][1])>=ymin
		and max(self.mesh.vectors[i][0][1],self.mesh.vectors[i][1][1],self.mesh.vectors[i][2][1])<=ymax]
		# and min(self.mesh.vectors[i][0][2],self.mesh.vectors[i][1][2],self.mesh.vectors[i][2][2])>0]

		points_3d = np.array(vectors_valid)
		# print(points_3d,points_3d.shape)

		xs = points_3d[:,:,0].flatten()
		ys = points_3d[:,:,1].flatten()
		zs = points_3d[:,:,2].flatten()

		# print(xs,xs.shape)
		Z = griddata((xs, ys), zs, (X, Y),method='linear')

		# 
		pt2tri =self.map_points2trianle(vectors_valid,xGrid,yGrid)


		for j in range(len(yGrid)):
			for i in range(len(xGrid)):
				
				if j%2==1:
					idx = len(xGrid)-1-i  
				else:
					idx = i
				if idx < len(xGrid)-1:
					pt1 = Point(xGrid[idx],yGrid[j])
					pt2 = Point(xGrid[idx+1],yGrid[j])
					# get corresponding triangle of this point
					if (pt1.x,pt1.y) in pt2tri: # the point:tri pair are exist
						ptz = self.calculate_z_v2(pt1,pt2tri[(pt1.x,pt1.y)])
					else:
						ptz = Z[j][idx]
					# ptz = self.calculate_z(xGrid[idx],yGrid[j],vectors_valid) ## first version, extremely slow, but high accuracy
					# ptz = Z[j][idx] ## scipy library, very coarse, low resolution
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
					if (pt1.x,pt1.y) in pt2tri: # the point:tri pair are exist
						ptz = self.calculate_z_v2(pt1,pt2tri[(pt1.x,pt1.y)])
					else:
						ptz  = Z[idx][j]
					# ptz = self.calculate_z(xGrid[j],yGrid[idx],vectors_valid)
					# ptz = Z[idx][j]
					## from left to right
					if ~np.isnan(ptz):
						if pt1.within(poly):
							edge_pts.append((pt1.x,pt1.y,ptz))
						if pt1.within(poly)==False and pt2.within(poly)==True:
							edge_pts.append(((pt1.x+pt2.x)/2.0,pt1.y,ptz))
							
						if pt1.within(poly)==True and pt2.within(poly)==False:
							edge_pts.append(((pt1.x+pt2.x)/2.0,pt1.y,ptz))
		print('3D tool-path generated!')
		return np.array(edge_pts),X,Y,Z

	def contour2path(self,path,fileName,):
		scale_factor = 0.16;
		offset_x = 0
		offset_y = 0
		offset_z = 1
		xyz = path
		coordsXYZ = xyz*scale_factor
		# print coordsXYZ,coordsXYZ.shape
		coordsXYZ[:,0] = coordsXYZ[:,0]+ offset_x# move the whole strucuture along X direction by a offset
		## Compute the path length and extrusion angle between each adjacent points
		## initialize parameters
		default_speed = 70; # mm/min
		user_input_angular_vel = 0.5; # mL/hr
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


		