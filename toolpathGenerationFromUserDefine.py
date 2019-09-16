import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
class toolpathGenerationFromUserDefine:
    def __init__(self,revol,dz,length,width,feed):
        # Parameter initiation
        # define square
        # # z incremental each revolution
        # dz = 0.2
        # #  revolution or layer number  
        # revol =3
        #  # length of the cubic 
        # length =5
        #  # width of the cubic
        # width =5
        #  # feed rate of each layer
        # feed =1
        self.revolution = revol
        self.dz = dz
        self.length = length
        self.width = width
        self.feed = feed

    def generate_tool_path(self):
        width = self.width
        revol = self.revolution
        dz = self.dz
        length = self.length
        feed = self.feed
        # Compute key points from the defined cube with continuous path
        key_pt_w = np.linspace(0,width,num=width/feed+1)
        key_pt_w_edge = [0]*(len(key_pt_w)*2)

        for i in range(1,2*len(key_pt_w)):
            idx = (i+1)//2
            if idx%2:
                key_pt_w_edge[i] = length
            else:
                key_pt_w_edge[i] = 0

        key_pt_w_inc = np.repeat(key_pt_w,2);
        key_pt_w_coor = np.concatenate(([key_pt_w_inc],[key_pt_w_edge]),axis=0)
        # print(key_pt_w_coor)
        key_pt_l = np.linspace(0,length,num=length/feed+1)
        key_pt_l_edge = [0]*(len(key_pt_l)*2)

        for i in range(1,2*len(key_pt_l)):
            idx = (i+1)//2
            if idx%2:
                key_pt_l_edge[i] = width
            else:
                key_pt_l_edge[i] = 0

        key_pt_l_inc = np.repeat(key_pt_l,2);
        key_pt_l_coor =np.concatenate(([key_pt_l_edge],[key_pt_l_inc]),axis=0)

        Ro = [[1,0,0],[0,1,0],[0,0,1]] # no change matrix
        Rl = [[1,0,0],[0,-1,length],[0,0,1]]# rot matrix to right
        Rw = [[-1, 0,width],[0,1,0],[0,0,1]]; # rot matrix to top
        Rlw =[[-1,0,width],[0,-1,length],[0,0,1]] # rot matrix to top right
        line_num = len(key_pt_w_coor[0]) # line number of each layer
        print(line_num)
        # continous tool path
        for i in range(1,revol+1):
            if i%2:
                xyz_add = key_pt_l_coor         
            else:
                xyz_add = key_pt_w_coor

            if i%4 == 1:
                R = Ro      
            else:
                x = xyz[:,1]
                y = xyz[:,0]
                if x[-1] == 0 and y[-1] >0: # Rw
                    R = Rw
                elif x[-1] > 0 and y[-1] == 0: # Rl 
                    R = Rl
                elif x[-1] > 0 and y[-1] > 0: # Rlw
                    R = Rlw
                elif x[-1] == 0 and y[-1] ==0:
                    R = Ro   
         
            aug = np.ones(len(xyz_add[0]),dtype='float')
            xyz_add_xy_ag = np.concatenate((xyz_add,[aug]),axis=0)
            xyz_aug = np.matmul(R,xyz_add_xy_ag).T

            zpt = (i-1)*dz*aug
            xyz_aug[:,2] = np.asarray(zpt)
            if i == 1:
                xyz = xyz_aug
            else:
                xyz = np.append(xyz,xyz_aug,axis=0)


        ## Calibration of the coordinates by timing a scale factor
        scale_factor = 0.16; #scale_factor is the Real_dimension/Set_dimension
        offset_x       = 2.0; # offset of the scaffold
        offset_y       = 0.0;
        offset_z       = 1; 

        # starting points p1 p2


        coordsXYZ = xyz*scale_factor
        coordsXYZ[:,0] = coordsXYZ[:,0]+ offset_x# move the whole strucuture along X direction by a offset
        startPts = [[0,0,0],[0,0,offset_z],[coordsXYZ[0,0], coordsXYZ[0,1] ,offset_z]]
        endPts = [[coordsXYZ[0,0], coordsXYZ[0,1] ,offset_z],[0,0,offset_z],[0,0,0]]
        coordsXYZ = np.concatenate((startPts,coordsXYZ,endPts),axis=0)
        # print(coordsXYZ)
        # print(xyz)
        # fig = plt.figure()
        # ax = fig.add_subplot(111,projection='3d')
        # plt.plot(coordsXYZ[:,0],coordsXYZ[:,1],coordsXYZ[:,2])
        # plt.show()


        #  Compute the path length and extrusion angle between each adjacent points
        #  initialize parameters
        default_speed_set = 6; # mm/s 
        speed_factor = 10;
        default_speed = default_speed_set*speed_factor;
        # speed_inc = 10;
        # compute the flowrate 
        user_input_angular_vel = 3; # mL/hr
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


        return coordsXYZ,Angle,Times,line_num
        # ## Convert the toolpath to G-code
        # # plot the tool-path in python
        # ## open the file
        # fileName ='path.nc'
        # fileIDx=open(fileName,'w');  #file where comands are appended
        # # value = ('X',coordsXYZ[0,0],coordsXYZ[1,0],coordsXYZ[2,0],Angle[0],default_speed)
        # formatSpec= 'X%7.6f Y%7.6f Z%7.6f A%7.6f F%7.6f;%7.6f\n';
        # ## initial the G-code file
        # fileIDx.write('G90\nG49\n')
        # fileIDx.write('G1 X0 Y0 Z0 F60\n')#mm/min
        # ## fprintf(fileIDx,'G1 X0 Y0 Z3')
        # ## add transion path in front and end 
        # # fileIDx.write(formatSpec %(0, 0 ,offset_z ,Angle[0] ,default_speed,0))
        # # fileIDx.write(formatSpec %(coordsXYZ[0,0], coordsXYZ[0,1] ,offset_z ,Angle[0] ,default_speed,0))
        # # fileIDx.write(formatSpec %(coordsXYZ[0,0], coordsXYZ[0,1] ,0 ,Angle[0] ,default_speed,0))
        # ### write down the coordinates one by one in format: G1 X Y Z A F\
        # for i in range(len(coordsXYZ)):
        #     fileIDx.write(formatSpec %(coordsXYZ[i,0], coordsXYZ[i,1] ,coordsXYZ[i,2] ,Angle[i] ,default_speed,Times[i]))

        # fileIDx.write(formatSpec %(coordsXYZ[i,0], coordsXYZ[i,1] ,offset_z ,Angle[i] ,default_speed,Times[i]))
        # fileIDx.write(formatSpec %(0, 0 ,offset_z ,Angle[i] ,default_speed,Times[i]))
        # fileIDx.write('G1 X0 Y0 Z0 F100\n')#mm/min
        # fileIDx.write('M05\nM02\n');
        # fileIDx.close()


# tool_path_for_print = toolpathGenerationFromUserDefine(revol=3,dz=0.1,length=10,width=10,feed=2)
# path,angles,times,line_num = tool_path_for_print.generate_tool_path()
# # plot the path 
# figure = Figure(figsize=(3,3), dpi=100) 
# a = figure.add_subplot(111,projection='3d')
# a.plot(path[:,0],-path[:,1],path[:,2])
# a.scatter(0,0,0)
# a.set_xlabel('X direction /mm')
# a.set_ylabel('Y direction /mm')
# a.set_zlabel('Z directions/mm')
# a.set_title(' *** Generated Dynamic Toolpath ***')
# # self.a.set_aspect('equal', 'box')
# xmin = min(path[:,0])-20;xmax = max(path[:,0])+50;
# ymin = min(path[:,1])-10;ymax = max(path[:,1])+10;
# zmin = min(path[:,2])-10;zmax = max(path[:,2])+10;
# a.set_xlim([xmin,xmax])
# a.set_ylim([ymin,ymax])
# a.set_zlim([zmin,zmax])
# plt.draw()
