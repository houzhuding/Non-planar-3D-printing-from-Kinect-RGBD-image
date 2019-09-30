"""
shape matching matrix

Generate matrix between two shape with soild transformation

"""

import numpy as np 
import random

def uniform(points,sample_num):

	x = points[:,0]
	y = points[:,1]

	xd =np.diff(x)
	yd = np.diff(y)
	dist = np.sqrt(xd**2+yd**2)
	u = np.cumsum(dist)
	u = np.hstack([[0],u])

	t = np.linspace(0,u.max(),sample_num)
	xn = np.interp(t, u, x)
	yn = np.interp(t, u, y)

	return np.transpose(np.append([xn],[yn],axis=0))

def ransac_ding(ucnt1,cnt2,sample_num):
	"""
	RANSAC to find homograph transformation between two outlines
	"""
	# fig = plt.figure(figsize=(10,10))
	# ax = fig.add_subplot(111)
	# ax.scatter(cnt1[:,0],cnt1[:,1],color='red')
	# ax.scatter(cnt2[:,0],cnt2[:,1],color='blue')

	# concave_hull_cnt2, edge_points = alpha_shape(cnt2, alpha=0.5)
	# poly2 = Polygon(cnt2)
	# actual_area = poly2.area
	# print(actual_area)
	pts_num = 4 
	H_found = 0
	thres = 2
	iterations = 0
	min_error = float('inf')
	# while iterations<10000:
	for i in range(sample_num):

		cnt1 = np.append(ucnt1[i:,:],ucnt1[:i,:],axis=0)
		# define the vector to store inliers' index of all random points
		pts_selected = []
		# define the vector to store inliners every iteration
		inliers = []
		# generate four random indices and select them from cnt1
		pts_order1 = (random.sample(range(0,len(cnt1)-1),4))
		pts_1 = cnt1[pts_order1,:]
		# generate four random indices and select them from cnt2
		pts_order2 = random.sample(range(0,len(cnt2)-1),4)
		pts_2 = cnt2[pts_order1,:]

		# Calculate the homograph matrix between eight points
		x1,y1 = pts_1[0,:]
		x2,y2 = pts_1[1,:]
		x3,y3 = pts_1[2,:]
		x4,y4 = pts_1[3,:]
		xp1,yp1 = pts_2[0,:]
		xp2,yp2 = pts_2[1,:]
		xp3,yp3 = pts_2[2,:]
		xp4,yp4 = pts_2[3,:]

		A=np.array([
		[-x1  ,-y1 , -1 ,  0 ,   0   , 0 ,  x1*xp1  , y1*xp1 ,  xp1],
 		[0  ,  0,    0 ,-x1  , -y1 , -1,   x1*yp1 ,  y1*yp1 ,  yp1],
		[-x2 , -y2 , -1 ,  0 ,   0  ,  0  , x2*xp2 ,  y2*xp2  , xp2],
 		[0  ,  0  ,  0 ,-x2  , -y2 , -1  , x2*yp2,  y2*yp2 ,  yp2],
		[-x3 , -y3 , -1  , 0  ,  0  ,  0 ,  x3*xp3 ,  y3*xp3,   xp3],
		[ 0 ,   0  ,  0 ,-x3,  -y3 , -1 ,  x3*yp3  , y3*yp3  , yp3],
		[-x4 , -y4  , -1 , 0  ,  0  ,  0 ,  x4*xp4 ,  y4*xp4 ,  xp4],
 		[0  ,  0   , 0  ,-x4 ,  -y4, -1  , x4*yp4 ,  y4*yp4   ,yp4]])

		U,S,V = np.linalg.svd(A)
		H_found = V[8,:].reshape(3,3)
		H_found = (1.0/H_found.item(8))*H_found
		# compute the outliners 
		cnt1_ag = np.array([cnt1[:,0],cnt1[:,1],[1]*len(cnt1)])
		tcnt1 =np.transpose(np.dot(H_found,cnt1_ag))
		tcnt1_xy =  np.transpose([np.divide(tcnt1[:,0],tcnt1[:,2]),np.divide(tcnt1[:,1],tcnt1[:,2])])
		error_= np.linalg.norm(tcnt1_xy-cnt2[:,0:2])
		if error_<=min_error:
			min_error = error_
			H_found_correct = H_found
			tcnt1_correct = tcnt1_xy[:,0:2]
			best_iter = iterations
			print(iterations,error_)
		# if error_ <= thres:
		# 	H_found_correct = H_found
		# 	tcnt1_correct = tcnt1_xy[:,0:2]
		# 	ax.scatter(tcnt1_xy[:,0],tcnt1_xy[:,1],color='green')
		# 	plt.show()
		# 	return H_found_correct,tcnt1_xy[:,0:2]
		iterations+=1
	print('Best fit', best_iter)
	# ax.plot(tcnt1_correct[:,0],tcnt1_correct[:,1],color='green')	
	# plt.show()
	return H_found_correct,tcnt1_correct