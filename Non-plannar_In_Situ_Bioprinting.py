"""In situ bioprinting control
	This program aims to control bioprinting 
	process by adding computer vision method
	This is initiated on 12/5/2018
	Author: Houzhu Ding
	This is Bioprinter Control GUI V8.0

    Copyright 2019, Houzhu Ding ,All rights reserved
"""

### Define a GUI to visualize the process ### 

import numpy as np
import cv2
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import sys
import serial
from tkinter import *
from PIL import Image,ImageTk 
import time 
import math
from queue import *
from threading import Thread, Condition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
from stl import mesh
from alpha_shape import alpha_shape
# from sklearn.cluster import KMeans

######## Load customized methods ########
from imgProcessMethods import *
from AdjustPrinterXYZA import *
from toolpathGenerationFromContour import *
from toolpathGenerationFromContour3D import *
from toolpathGenerationFromUserDefine import *
from shape_match import *

######## The main class ################
class BioprinterControl():
    def __init__(self,master):
        self.master = master
        self.frame = Frame(self.master)

        Label(master, text="IMAGE GUIDED BIOPRINTING CONTROL PANEL Ver8.0").grid(row=0,columnspan=10)
        #self.LL0 = Label(master, text ="--------------------------").grid(rowspan=2,columnspan=2)
        ### variables initialization
        self.portNum = 'COM3'
        self.camNum = 0
        self.camNumNeedle = 1
        self.key_pts = []
        self.mouse_click = []
        self.pathFile = []
        self.s = None
        self.speed = 'F100'
        self.XYZA = AdjustPrinterXYZA(self.s,'0','0',self.speed)
        self.ImgProcessTool = ImgProcessMethods()
        self.cx = 5 #100
        self.cy = 5#10
        self.cw = 630#200 
        self.ch = 470#100
        self.cycle_index = -1
        self.line_index = -1
        self.lineNum = 10 

        # variables related to auto-calibration 
        self.show_botton = 0 ## show transformed view (from perspective to top)
        self.needle_location = [] ## needle position,user point on image 
        self.rect = []

        # printer initial parameters
        self.x_in=0
        self.y_in=0
        self.z_in=0
        self.start_tracking = 1
        self.path3d = []
        self.contour_model = np.array([0,0])
#################### Open camera Button ################################
        self.openCam = Button(master,width = 10,text = 'Open Camera',command = lambda:self.initializeCameras(self.camNum,self.camNumNeedle))
        self.openCam.grid(row = 1,column = 1)  
#################### Calibrate Button ################################
        Button(master,width = 10,text = 'Connect',command = lambda:self.Connect(self.portNum)).grid(row =2,column = 1)    
#################### Disconnect  ################################
        Button(master,width = 10,text = 'Disconnect',command = self.Disconnect).grid(row = 4,column = 1) 
#################### Close camera Button ################################
        Button(master,width = 10,text = 'Monitoring',command = self.StartMonitoring).grid(row = 3,column = 1)       
#################### Close camera Button ################################
        Button(master,width = 20,text = 'Extract Contour',command = self.ExtractContour).grid(row = 1,column = 7)                 
#################### Print the contour  ################################
        Button(master,width = 20,text = 'Print Contour',command = self.PrintContourThread).grid(row = 2,column = 7)      
#################### Open Path File  ################################
        Button(master,width = 20,text = 'Load Path',command = self.LoadPath).grid(row = 3,column = 7)   
        
#################### Set Home  ################################
        Button(master,width = 15,text = 'Set Zero(A)',command = self.XYZA.SetZerosA).grid(row = 4,column = 3) 
        Button(master,width = 15,text = 'Set Zeros(XYZ)',command = self.XYZA.SetZerosXYZ).grid(row = 4,column = 4)    
        Button(master,width = 15,text = 'Go Zeros',command = self.XYZA.GoToZeros).grid(row = 4,column = 5)  
        Button(master,width = 15,text = 'Go to XYZ',command = lambda:self.XYZA.GoToXYZ(self.x_in,self.y_in,self.z_in)).grid(row = 4,column = 6)  
#################### Calibrate  ################################

        Button(master,width = 20,text = 'Draw  Toolpath',command = self.DrawDanamicToolPath).grid(row = 5,column = 7) 
        Button(master,width = 20,text = 'Print Toolpath',command = self.PrintDanamicToolPath).grid(row = 6,column = 7)    
   
#################### Show Top view  #########################   
        Button(master,width= 20,text = 'Show Top-view',command = self.ShowTopView).grid(row = 7,column=7)
#################### Select Needle Position  #########################   
        Button(master,width= 20,text = 'Select Needle',command = self.SelectNeedle).grid(row = 7,column=6)

        Button(master,width= 20,text = 'Start Follow',command = self.StartFollow).grid(row = 5,column=6)
        Button(master,width= 20,text = 'Stop  Follow',command = self.StopFollow).grid(row = 6,column=6)


        acol = 2; xcol = 3;  ycol = 4; zcol = 5
#################### Control the Printer extrusion axis  ##############################
        Label(master,text ="Extrusion flowrate (1 ml/h)").grid(row = 1,column = acol,padx=15, pady=5)

        self.x_plus = Button(master,text = 'Extrude',command = lambda:self.XYZA.A_plus(self.incA.get()))
        self.x_plus.grid(row = 2,column = acol,padx=15, pady=5)

        self.x_minus = Button(master,text = 'Unload',command = lambda:self.XYZA.A_minus(self.incA.get()))
        self.x_minus.grid(row = 3,column = acol,padx=15, pady=5)           
#################### Control the Printer x axis  ##############################
        Label(master,text ="X direction").grid(row = 1,column = xcol,padx=15, pady=5)

        self.x_plus = Button(master,text = ' X + ',command = lambda:self.XYZA.X_plus(self.incXYZ.get()))
        self.x_plus.grid(row = 2,column = xcol,sticky=N+S+E+W)

        self.x_minus = Button(master,text = ' X - ',command = lambda:self.XYZA.X_minus(self.incXYZ.get()))
        self.x_minus.grid(row = 3,column = xcol,sticky=N+S+E+W)       

#################### Control the Printer y axis  ##############################
        Label(master,text ="Y direction").grid(row = 1,column = ycol,padx=15, pady=5)

        self.y_plus = Button(master,text = ' Y + ',command = lambda:self.XYZA.Y_plus(self.incXYZ.get()))
        self.y_plus.grid(row = 2,column = ycol,sticky=N+S+E+W)

        self.y_minus = Button(master,text = ' Y - ',command = lambda:self.XYZA.Y_minus(self.incXYZ.get()))
        self.y_minus.grid(row = 3,column = ycol,sticky=N+S+E+W)
       
#################### Control the Printer z axis  ##############################
        Label(master,text ="Z direction").grid(row = 1,column = zcol,padx=15, pady=5)

        self.z_plus = Button(master,text = ' Z + ',command = lambda:self.XYZA.Z_plus(self.incXYZ.get()))
        self.z_plus.grid(row=2,column=zcol,sticky=N+S+E+W)

        self.z_minus = Button(master,text = ' Z - ',command = lambda:self.XYZA.Z_minus(self.incXYZ.get()))
        self.z_minus.grid(row=3,column=zcol,sticky=N+S+E+W)

#################### Define the increment of each axis  ##########################
        Label(master,text ="Axis/Degree Inc(mm/degree)").grid(row = 1,column = zcol+1)

        self.incXYZ = StringVar(None);self.incXYZ.set("0.5")
        self.SelIncXYZ=OptionMenu(master,self.incXYZ,'0.1','0.2','0.3','0.4','0.5','1','2','3').grid(row = 2,column = zcol+1,padx=5, pady=5)

        self.incA = StringVar(None);self.incA.set("1")
        self.SelIncA=OptionMenu(master,self.incA,'1','2','3','4','5','6','7','8').grid(row = 3,column = zcol+1,padx=5, pady=5)  


        # Label(master,text ="Image to real (pixels/mm)").grid(row=1,column=8)

        # scaleImg2Real = StringVar(None);scaleImg2Real.set("12.84")
        # self.EntryBoxImgReal = Entry(master,width=5,textvariable = scaleImg2Real).grid(row=2,column=8)

        # Label(master,text ="Camera to needle x,y (mm)").grid(row=3,column=8)

        # offsetX = StringVar(None);offsetX.set("35")
        # self.EntryBoxOffsetX = Entry(master,width=5,textvariable=offsetX).grid(row=4,column=8)        
        # offsetY = StringVar(None);offsetY.set("3")
        # self.EntryBoxOffsetY = Entry(master,width=5,textvariable=offsetY).grid(row=5,column=8) 
#################### Define the scaffold parameters  ##########################

        Label(master,text="PRINTER PARAMETERS").grid(row=1,column=8,columnspan=5)

        Label(master,text ="Layer number").grid(row=2,column=8)
        layerNumTextV = StringVar(None);layerNumTextV.set("2")
        self.EntryBoxLayerNumber = Entry(master,width=5,textvariable = layerNumTextV)
        self.EntryBoxLayerNumber.grid(row=2,column=9)

        Label(master,text ="Length (x) of square (mm)").grid(row=3,column=8)
        edgeLength = StringVar(None);edgeLength.set("10")
        self.EntryBoxLength = Entry(master,width=5,textvariable=edgeLength)
        self.EntryBoxLength.grid(row=3,column=9)  

        Label(master,text ="Length (y) of square (mm)").grid(row=4,column=8)
        edgeWidth = StringVar(None);edgeWidth.set("10")
        self.EntryBoxWidth = Entry(master,width=5,textvariable=edgeWidth)
        self.EntryBoxWidth.grid(row=4,column=9) 

        Label(master,text ="Inter filament distance(mm)").grid(row=5,column=8)
        interFilament = StringVar(None);interFilament.set("2")
        self.EntryBoxSf = Entry(master,width=5,textvariable=interFilament)
        self.EntryBoxSf.grid(row=5,column=9) 

        Label(master,text ="Layer height (mm)").grid(row=6,column=8)
        initialDz = StringVar(None);initialDz.set("0.1")
        self.EntryBoxDz = Entry(master,width=5,textvariable=initialDz)
        self.EntryBoxDz.grid(row=6,column=9) 

#################### Define/Amend printing parameters ######################

        Label(master,text="Measured Speed(mm/s)").grid(row=2,column=10,columnspan=3)
        self.tSpeed = StringVar();self.tSpeed.set("120")
        self.EntryBoxtSpeed = Entry(master,width=5,textvariable = self.tSpeed).grid(row=3,column=11,sticky=N+S) 
        self.v_plus = Button(master,text = ' V + ',command = lambda:self.V_plus(self.tSpeed.get())).grid(row=3,column=10,sticky=W+E+N+S)
        self.v_minus = Button(master,text = ' V - ',command = lambda:self.V_minus(self.tSpeed.get())).grid(row=3,column=12,sticky=W+E+N+S)

        self.LflowRate = Label(master,text ="Flowrate (ml/h)").grid(row=4,column=10,columnspan=3)

        self.flowRateV = StringVar(None);self.flowRateV.set("3")
        self.EntryBoxflowRate = Entry(master,width=5,textvariable = self.flowRateV).grid(row=5,column=11,sticky=N+S)
        self.f_plus = Button(master,text = ' F + ',command = lambda:self.F_plus(self.flowRateV.get())).grid(row=5,column=10,sticky=W+E+N+S)
        self.f_minus = Button(master,text = ' F - ',command = lambda:self.F_minus(self.flowRateV.get())).grid(row=5,column=12,sticky=W+E+N+S)


        Label(master,text ="Feed rate (mm)").grid(row=6,column=4)
        feedTextV = StringVar(None);feedTextV.set("1")
        self.EntryBoxFeed = Entry(master,width=5,textvariable = feedTextV)
        self.EntryBoxFeed.grid(row=6,column=5)

        ### show speed on each single cycle and single segments
        Label(master,bg="red",text='Real time speed record(mm/s) ').grid(row=6,column=10,columnspan=3)
        Label(master,bg="red",text='Cycle index    Line index ').grid(row=7,column=10,columnspan=3)

        self.cycle_list = Listbox(master,width=10, height=10,bg='white',selectmode='SINGLE')
        self.cycle_list.grid(row = 8,column=10,columnspan=1,rowspan=3,sticky=W+E+N+S)
        self.cycle_list.bind("<<ListboxSelect>>",self.ClickToChangeSpeed_CycleNum)

        self.speed_list = Listbox(master,width=10, height=10,selectmode='SINGLE')
        self.speed_list.grid(row = 8,column=11,columnspan=2,rowspan=3,sticky=W+E+N+S)
        self.speed_list.bind("<<ListboxSelect>>",self.ClickToChangeSpeed_LineNum)
        self.speed_list.bind("<Double-Button-1>",self.ChanageSpeed)

#################### Show Camera Frame  ##############################
        self.w = Label(master,height=480,bg="white")
        self.w.grid(row=8,rowspan=2,columnspan=6,sticky=W+E+N+S)

        self.c = Canvas(master,bg="white")
        self.c.config(width=320, height=240)
        self.c.grid(row = 8,column = 6,columnspan = 2,sticky=W+E+N+S)

        self.c_needle = Canvas(master,bg="white")
        self.c_needle.config(width=320, height=240)
        self.c_needle.grid(row = 8,column = 8,columnspan = 2,sticky=W+E+N+S)        
#################### Show Plotted Contour  ##############################     
        self.f = Canvas(master,bg="white")
        self.f.config(width=320, height=240)
        self.f.grid(row = 9,column = 6,columnspan = 4,sticky=W+E+N+S)     
        self.figure = Figure(figsize=(3,3), dpi=100) 
        self.a = self.figure.add_subplot(111,projection='3d')

        self.figure2 = Figure(figsize=(3,3), dpi=100) 
        self.b = self.figure2.add_subplot(111,projection='3d')

        self.plot_initially()
        # self.list = Listbox(master,width = 30, height = 500)
        # self.list.grid(row = 11,column=3,columnspan =  1)
#################### Registration Manually  ##############################     
        self.sliderbarX = Scale(master,from_=-20,to=20,orient=HORIZONTAL,label='X',resolution = 1)
        self.sliderbarX.grid(row=6,column=1,columnspan=1)
        self.sliderbarX.set(0)
        self.sliderbarY = Scale(master,from_=-20, to=20,orient=HORIZONTAL,label='Y',resolution = 1)
        self.sliderbarY.grid(row =6,column = 2,columnspan = 1)
        self.sliderbarY.set(0)
#################### Define the threshold  ##########################
        # thresholdText = StringVar()
        # thresholdText.set("Image to real (pixels/mm)")
        # self.L9 = Label(master,textvariable =thresholdText)
        # self.L9.grid(row = 3,column = 6,padx=10, pady=5)

        # threshold_num = StringVar(None)
        # threshold_num.set("150")
        # self.EntryBoxThres = Entry(master,width=5,textvariable = threshold_num)
        # self.EntryBoxThres.grid(row = 4,column = 6,padx=5, pady=5)
        self.sliderbarThres = Scale(master,from_ = 1,  to=20,orient=HORIZONTAL,label='Scale the contour')
        self.sliderbarThres.grid(row = 6,column = 3,columnspan = 1)
        self.sliderbarThres.set(10)

        self.cycle_list.insert(END,'cycle'+' : '+str(1))
        self.cycle_list.insert(END,'cycle'+' : '+str(2))

        self.Speeds = [10]*self.lineNum

        for i in range(10):
            self.speed_list.insert(END,'line '+str(i)+'  : '+str(self.Speeds[i]))
        return

    def plot_initially(self):
        self.a.scatter(0,0,0)
        self.a.set_xlabel('X direction')
        self.a.set_ylabel('Y direction')
        self.a.set_title('Toolpath planning')
        # self.a.set_aspect('equal', 'box')
        self.cntPlot = FigureCanvasTkAgg(self.figure,master=self.f)    
        self.cntPlot.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self.cntPlot.draw()

        self.b.scatter(0,0,0)
        self.b.set_xlabel('X direction')
        self.b.set_ylabel('Y direction')
        self.b.set_title('Display STL file')
        self.stlPlot = FigureCanvasTkAgg(self.figure2,master=self.c)    
        self.stlPlot.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self.stlPlot.draw()

#######################################################################################
#######################################################################################
#################    Start to use camera to control 3D printer        #################
#######################################################################################
#######################################################################################


############################### imaging/camera part ###################################
    def initializeCameras(self,camNum1,camNum2):
        self.cap1 = cv2.VideoCapture(camNum1)
        # Check if camera opened successfully
        if (self.cap1.isOpened()== False): 
            print("Error opening video stream or file")
        count = 0
        self.closeCamSign = 1
        self.contour = []
        while(self.closeCamSign):
            # Capture frame-by-frame
            ret, frame = self.cap1.read()
            # Our operations on the frame come here
            self.rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Display the resulting frame
            cv2.imshow('frame',self.rgb)
            cv2.setMouseCallback('frame',self.ClickAndCrop)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if count > 10:
                break
            count += 1
        cv2.destroyAllWindows()
    def SelectNeedle(self,):
        if self.img_crop is not None:
            
            self.rgb_needle = self.img_crop.copy()
            cv2.namedWindow('Select Needle')
            cv2.setMouseCallback('Select Needle',self.draw_needle_position)

            while(1):

                cv2.imshow('Select Needle',self.rgb_needle)
                if len(self.needle_location)==1 and self.needle_updated ==1:
                    print('Needle position',self.needle_location[0])
                    self.needle_updated = 0
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break

            cv2.destroyAllWindows()
        else:
            print('There are no live images!')

    def draw_needle_position(self,event,x,y,flags,param):
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.needle_location = []
            self.rgb_needle = self.img_crop.copy()
            cv2.circle(self.rgb_needle,(x,y),3,(255,0,0),-1)
            self.needle_location.append((x,y))
            self.needle_updated = 1

    def ShowTopView(self):
        if self.show_botton == 1:
            self.show_botton = 0
        else:
            self.show_botton = 1
    def callback_needle_select(self,event):
        pass
        # if self.show_botton == 0:
        #     print ("Clicked at ", event.x,event.y)
        #     self.needle_location.append((int(event.x),int(event.y-20)))   
        # else:
        #     print("Select from perspective view!")
    def callback(self,event):
        print ("Clicked at ", event.x,event.y-40)
        # transfer to printer coordinate
        # real_scale_factor = 1/12.84
        # machine_scale_factor = 1/6.2
        # offsetX = float(self.EntryBoxOffsetX.get())
        # offsetY = float(self.EntryBoxOffsetY.get())
        # xLoc = ((event.x -320)*real_scale_factor-offsetX)*machine_scale_factor
        # yLoc = ((event.y-240)*real_scale_factor-offsetY)*machine_scale_factor
        # print ("Printer location at ", xLoc,yLoc)
        # color = 'red'
        # linewidth = 3
        # self.mouse_click.append((xLoc,yLoc))
        self.key_pts.append((int(event.x),int(event.y-40)))
        # if self.s is not None:
        #     l = 'G92 X0 Y0 Z0 A0\n'
        #     print 'Sending: ' + l,
        #     self.s.write(l + '\n') # Send g-code block to grbl
        #     grbl_out = self.s.readline() # Wait for grbl response with carriage return
        #     print ' : ' + grbl_out.strip()  
        #     l = 'G90\nG1 X'+str(xLoc)+'Y'+str(yLoc)+'F50'
        #     print 'Sending: ' + l,
        #     self.s.write(l + '\n') # Send g-code block to grbl
        #     grbl_out = self.s.readline() # Wait for grbl response with carriage return
        #     print ' : ' + grbl_out.strip()
        # r = 5
        # self.w.create_oval([120-r,120+r,160-r,160+r],fill = 'black',outline = "")             

            # self.w.delete("all")
            # self.w.create_line(self.key_pts[-2][:],self.key_pts[-1][:],fill = color,width= linewidth)
            # self.w.update()
        # When everything done, release the capture
    def LiveCamera(self,camNum,cap,canvas_):
        self.closeCamSign = 1
        count = 0
        # self.cropPos = []
        # self.cropping = False
        # self.id = None
        # cap = cv2.VideoCapture(camNum)
        if camNum == 1:
            cap.set(3,160)
            cap.set(4,120)
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        while(self.closeCamSign):  
            # Capture frame-by-frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = cap.read()
            if frame is not None:
                self.img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.img_crop = self.img_rgb[self.cy:self.cy+self.ch, self.cx:self.cx+self.cw]
            else:
                continue

            if len(self.key_pts)==2:
                self.cx = int(min(self.key_pts[0][0],self.key_pts[1][0]))
                self.cy = int(min(self.key_pts[0][1],self.key_pts[1][1]))
                self.cw = int(math.fabs(self.key_pts[0][0]-self.key_pts[1][0]))
                self.ch = int(math.fabs(self.key_pts[0][1]-self.key_pts[1][1]))
                # print self.cx,self.cy,self.cw,self.ch

                self.img_crop = self.img_rgb[self.cy:self.cy+self.ch, self.cx:self.cx+self.cw]
                
                # print self.key_pts[-2][:],self.key_pts[-1][:]
                # draw a rectangle on the image
                cv2.rectangle(self.img_rgb,self.key_pts[-2][:],self.key_pts[-1][:],(0,255,0),3)
                self.key_pts = []

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.img_rgb))
            if camNum == 0 and self.img_crop is not None:
                self.w.imgtk= self.photo
                self.w.configure(image = self.photo)
                self.w.image_cache = self.photo
                self.w.bind("<Button-1>",self.callback)
                K=2
                scale_fac =1
                kernal_num =10
                # if self.sliderbarKmean.get():
                #     K = self.sliderbarKmean.get()
                # else:
                #     K = 3
                # scale_fac = self.sliderbarScale.get()
                # kernal_num = self.sliderbarThres.get()

                ### get contour
                # img_rgb_out,contour_out,roi_out,kmeanLines_small_tk_out,final_pts=  \
                #     self.ImgProcessTool.ContourExtractKmean(self.img_crop,K,scale_fac,kernal_num)
                
                 
                ### Calibrate 2D object following perspective view
                # print(self.needle_location)
                roi_out,needle_loc_trans,contour_out,rect,contour_out_model,mask_bk = self.ImgProcessTool.ContourExtractArUcoMarker(self.img_crop,self.show_botton,self.needle_location)
                # roi_out = self.ImgProcessTool.ContourExtractSVM(self.img_crop,count)
                # self.contour =  contour_out
                self.contour = contour_out
                self.contour_model = contour_out_model
                self.rect = rect
                self.model_mask_bk = mask_bk
                ##show needle region ##
                # if len(self.needle_location)==1:
                #     self.needle_location = []

                self.roi = roi_out
                # canvas_.image = self.edge_hsv
                # count+=1
                self.c_needle.image = roi_out
                self.c_needle.create_image(160, 120, image = roi_out) 
                if self.show_botton==0:
                    self.c_needle.bind("<Button-1>",self.callback_needle_select)

                # canvas_.create_image(320, 240, image = self.edge_hsv) 
                # canvas_.bind("<Button-1>",self.callback) 

            # if final_pts is not None:
            #     self.a.clear()
            #     x = range(len(final_pts))
            #     self.a.scatter(x,final_pts)
            #     self.a.set_aspect('equal', 'box')
            #     self.cntPlot.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
            #     self.cntPlot.show()

            # else:
            #     canvas_.image = self.img_rgb
            #     canvas_.create_image(160, 120, image = self.img_rgb) 
            #     canvas_.bind("<Button-1>",self.callback) 
        cap.release()  

########################## Imaging aided 3D printing part ##########################

    ### Calibrate the extrusion automatically ###
    def Calibrate(self):
        # K = self.sliderbarKmean.get()
        # c_needle_img = self.ImgProcessTool.FindNeedleAndPlatform(self.img_rgb,K)  
        # self.c_needle.image = c_needle_img
        # self.c_needle.create_image(160, 120, image = c_needle_img) 
        pass

############################### 3D printing part ###################################
    ### sending dynamically generated tool-path to printiner (speed flowrate are changable)
    def DrawDanamicToolPath(self):
        ## Draw toolpath from user defined
        # input_layer_num = int(self.EntryBoxLayerNumber.get())
        # input_dz        = float(self.EntryBoxDz.get())
        # input_length = float(self.EntryBoxLength.get())
        # input_width  = float(self.EntryBoxWidth.get())
        # intpu_feed   = float(self.EntryBoxSf.get())

        # self.tool_path_for_print = toolpathGenerationFromUserDefine(
        #     revol=input_layer_num,
        #     dz=input_dz,
        #     length=input_length,
        #     width=input_width,
        #     feed=intpu_feed)
        # self.path,self.Angles,self.Times,self.lineNum = self.tool_path_for_print.generate_tool_path()

        ### Initialize the speed and flowrate for whole path 
        self.speed =  float(self.tSpeed.get())
        self.Speeds = [self.speed]*len(self.path)
        self.flowrate = float(self.flowRateV.get())
        self.flowRates = [self.flowrate]*(len(self.path)-3)+ [-self.flowrate]*3
        self.flowRates[0] = 0

        def plot_path():
            ### clean the listbox
            self.speed_list.delete(0,END)
            self.cycle_list.delete(0,END)

            ### plot  the tool-path in red if executed...
            for i in range(len(self.path)-1):
                self.a.plot([self.path[i,0],self.path[i+1,0]],
                            [self.path[i,1],self.path[i+1,1]],
                            [self.path[i,2],self.path[i+1,2]],
                            color='r', linestyle='-', linewidth=2)
                self.cntPlot.draw()
                self.speed_list.insert(END,'line '+str(i+1)+' : '+str(self.Speeds[i])+' mm/s')
                if((i-5)%self.lineNum==0):
                    cycle_number = int((i-5)/self.lineNum)+1
                    self.cycle_list.insert(END,'layer'+' : '+str(cycle_number))
                print('Drawing '+str(i+1)+'th line...')
                # time.sleep(self.Times[i])
        # plot the tool path if there are any

        try :
            self.a.cla()
            self.a.scatter(0,0,0)
            self.a.set_xlabel('X direction /mm')
            self.a.set_ylabel('Y direction /mm')
            self.a.set_zlabel('Z directions/mm')
            self.a.set_title(' *** Generated Dynamic Toolpath ***')
            # self.a.set_aspect('equal', 'box')
            
            xmin = min(self.path[:,0]);xmax = max(self.path[:,0]);
            ymin = min(self.path[:,1]);ymax = max(self.path[:,1]);
            zmin = min(self.path[:,2]);zmax = max(self.path[:,2]);
            # xmin_ = 
            self.a.set_xlim([xmin,xmax])
            self.a.set_ylim([ymin,ymax])
            self.a.set_zlim([zmin,zmax])
            self.cntPlot.draw()
        
            plot_path()
        except Exception as e:
            print(" No path ready!")

    def PrintDanamicToolPath(self):

        ### record video for future processing
        w = self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH);
        h = self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT); 
        # Define the codec and create VideoWriter object
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
        
        input_layer_num = int(self.EntryBoxLayerNumber.get())
        input_dz        = float(self.EntryBoxDz.get())
        input_length = float(self.EntryBoxLength.get())
        input_width  = float(self.EntryBoxWidth.get())
        intpu_feed   = float(self.EntryBoxSf.get())

        self.tool_path_for_print = toolpathGenerationFromUserDefine(
            revol=input_layer_num,
            dz=input_dz,
            length=input_length,
            width=input_width,
            feed=intpu_feed)

        self.path,self.Angles,self.Times,self.lineNum = self.tool_path_for_print.generate_tool_path()

        self.a.cla()
        self.a.plot(self.path[:,0],self.path[:,1],self.path[:,2],color='blue')
        self.cntPlot.draw()
        ### Initialize the speed and flowrate for whole path 
        self.speed =  float(self.tSpeed.get())
        self.Speeds = [self.speed]*len(self.path)
        self.flowrate = float(self.flowRateV.get())
        self.flowRates = [self.flowrate]*(len(self.path)-3)+ [-self.flowrate]*3
        self.flowRates[0] = 0
        print(self.flowRates)

        # comput the path speed and flowrate according to the x,y,z path 
        coordsXYZ = self.path # first point is (0,0,0)
        
        ### write  the coordinates one by one in format: G1 X Y Z A F\
        formatSpec= 'X%7.6f Y%7.6f Z%7.6f A%7.6f F%7.6f';
        def show_path():
            ### clean the listbox
            self.speed_list.delete(0,END)
            self.cycle_list.delete(0,END)

            ### insert the current line executed...
            for i in range(len(self.path)-1):
                self.speed_list.insert(END,'line '+str(i+1)+' : '+str(self.Speeds[i])+' mm/s')
                if((i-5)%self.lineNum==0):
                    cycle_number = int((i-5)/self.lineNum)+1
                    self.cycle_list.insert(END,'layer'+' : '+str(cycle_number))
                print('Drawing '+str(i+1)+'th line...')
                time.sleep(self.Times[i])
        def print_path():
            ### Print the tool-path
            ## Initial the G-code sending to tinyG
            self.s.write(('G90\nG49\n').encode())
            self.s.write(('G1 X0 Y0 Z0 F60\n').encode())#mm/min
            for i in range(1,len(coordsXYZ)):
                cur_path = np.linalg.norm(coordsXYZ[i-1,:]-coordsXYZ[i,:])
                speed = self.Speeds[i]
                ttime = cur_path/speed
                self.Times[i-1]=ttime*60
                angle = ttime*self.flowRates[i]
                if  i< len(coordsXYZ)-2:
                    self.Angles[i] = self.Angles[i-1]-angle
                else:
                    self.Angles[i] = self.Angles[i-1]+3*angle
                # form the command
                cmd = formatSpec %(coordsXYZ[i,0], coordsXYZ[i,1] ,coordsXYZ[i,2] ,self.Angles[i] ,speed)
                self.s.write((cmd+'\n').encode())
                time.sleep(ttime*60)
                print('Printing the '+str(i)+'th line...')
            # self.s.write(('G1 X0 Y0 Z0 F100\n').encode())#mm/min
            self.s.write(('M05\nM02\n').encode());
            
            print('Printing has finished!\n')
            time.sleep(2)
            ### Set current location as degree==0
            self.XYZA.SetZerosA()
            size =len(self.Angles)
            self.Angles = [0]*size
        ### define two threads to plot and print the path simultaneously
        Thread(target=show_path).start()
        Thread(target=print_path).start()

    def ClickToChangeSpeed_CycleNum(self,evt1):
            _widget = evt1.widget
            selection_cycle=self.cycle_list.curselection()
            # print(selection[0])
            if selection_cycle:
                _value = self.cycle_list.get(selection_cycle[0])
                print('Cycle number selected:',_value[8:])
                self.cycle_index = int(_value[8:])

    def ClickToChangeSpeed_LineNum(self,evt2):
            widget = evt2.widget
            selection_speed=self.speed_list.curselection()
            if selection_speed:
                _value = self.speed_list.get(selection_speed[0])
                print('Line number selected:',_value[5:8])
                self.line_index = int(_value[5:8])

    def ChanageSpeed(self,evt):
        self.top = Toplevel(self.master,height=20,width=20)
        self.top.config()
        idx = (self.cycle_index-1)*10+self.line_index
        Label(self.top,text='Speed(mm/s)'+str(self.Speeds[idx])).pack()
        self.NewSpeedEntryBox = Entry(self.top)
        self.NewSpeedEntryBox.pack()
        Button(self.top,text='Update',command=self.update_speed_button).pack()
    
    def update_speed_button(self):
        new_speed = self.NewSpeedEntryBox.get()
        speed_idx = idx = (self.cycle_index-1)*10+self.line_index
        self.Speeds[speed_idx] = new_speed
        self.speed_list.delete(self.line_index)
        self.speed_list.insert(self.line_index,'line '+str(self.line_index)+'  : '+str(self.Speeds[speed_idx]))
        self.top.destroy()

    def ShowStlModel(self,):
        pass
    def ExtractContour3D(self):
        ### Extract 3D contour and generate toolpath ###
        # define the countour of the ROI
            xmin = 0 
            xmax = 20
            ymin = 0
            ymax = 20

            feed = float(self.EntryBoxFeed.get())
            step = feed
            if self.contour is None:
                xy_region = [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax],[xmin,ymin]]
            else:
                xy_region = self.contour
                cnt_center = np.mean(xy_region,axis=0)
                hand_mesh = mesh.Mesh(self.hand_mesh.data.copy())
                dx = self.sliderbarX.get()
                dy = self.sliderbarY.get()
                # hand_mesh.x = -hand_mesh.x #- cnt_center[0]+dx
                # hand_mesh.y = -hand_mesh.y #- cnt_center[1]+dy

                if self.hand_mesh is not None:
                    ### get the contour of the model by projection
                    points_3d = np.array(self.hand_mesh.vectors)
                    xs = points_3d[:,:,0].flatten()
                    ys = points_3d[:,:,1].flatten()
                    zs = points_3d[:,:,2].flatten()
                    stl_xy_points = np.transpose(np.array([xs,ys]))
                    ### find the boundary of the stl file
                    concave_hull, edge_points = alpha_shape(stl_xy_points, alpha=0.8)
                    stl_boundary =np.array(list(concave_hull.boundary.coords))

                    x = stl_boundary[:,0]
                    y = stl_boundary[:,1]
                    self.b.scatter(x,y,color='blue')
                    self.stlPlot.draw()
            
                    ### compute the transformation matrix between stl and image
                    self.H = []
                    if self.contour_model is not None:
                        x_cm =self.contour_model[:,0]
                        y_cm =self.contour_model[:,1]

                        sample_num = 5000

                        cnt1 = np.append(stl_boundary,[stl_boundary[0,:]],axis=0)
                        cnt2 = np.append(self.contour_model,[self.contour_model[0,:]],axis=0)

                        ### Equidistant the contours by a number
                        ucnt1 = uniform(cnt1,sample_num)
                        ucnt2 = uniform(cnt2,sample_num)
                        # cnt1 = cnt1[::-1]
                        self.H,boundary_c = ransac_ding(ucnt1,ucnt2,sample_num) 
                        hand_mesh_new = mesh.Mesh(hand_mesh.data.copy())
                        self.b.scatter(x_cm,y_cm,color='red')
                        self.b.scatter(boundary_c[:,0],boundary_c[:,1])

                    if self.H is not None:
                        for i in range(3):
                            hand_mesh_ag = np.array([hand_mesh.x[:,i],hand_mesh.y[:,i],[1]*len(hand_mesh.x)])
                            hand_mesh_xyz =np.transpose(np.dot(self.H,hand_mesh_ag))
                            # hand_mesh_xy =  np.transpose([np.divide(hand_mesh_xyz[:,0],hand_mesh_xyz[:,2]),
                            #     np.divide(hand_mesh_xyz[:,1],hand_mesh_xyz[:,2])])
                            hand_mesh_new.x[:,i] = np.divide(hand_mesh_xyz[:,0],hand_mesh_xyz[:,2])
                            hand_mesh_new.y[:,i] = np.divide(hand_mesh_xyz[:,1],hand_mesh_xyz[:,2])
                    hand_mesh_new.z = 2*hand_mesh.z #-(min(hand_mesh.z[0]))

                    veclen = len(hand_mesh_new.vectors)
                    vectors_valid = hand_mesh_new.vectors
                    points_3d = np.array(vectors_valid)
                    xs = points_3d[:,:,0].flatten()
                    ys = points_3d[:,:,1].flatten()
                    zs = points_3d[:,:,2].flatten()
                    stl_contour = np.array([xs,ys,zs])

                    self.b.cla()
                    self.b = self.figure2.add_subplot(111)
                    self.b.plot(cnt1[:,0],cnt1[:,1])
                    self.b.plot(stl_contour[0,:],stl_contour[1,:])
                    self.b.plot(self.contour[:,0],self.contour[:,1])
                    self.b.plot(self.contour_model[:,0],self.contour_model[:,1])
                    self.b.set_aspect('equal', 'box')
                    # ### re-draw the STL file
                    # self.b.cla()
                    # scale = hand_mesh.points.flatten(-1)
                    # # print(scale)
                    # self.b.auto_scale_xyz(scale, scale, scale)
                    # self.b.add_collection3d(art3d.Poly3DCollection(hand_mesh.vectors))
                    # # self.a.set_aspect('equal', 'box')
                    self.stlPlot.draw()


                    toolPathObj3D = toolpathGenerationFromContour3D(np.array(xy_region),hand_mesh_new,step,feed)
                    key_pts_3d,X,Y,Z = toolPathObj3D.scanHorizental()

                    # self.b.plot(key_pts_3d[:,0],key_pts_3d[:,1],key_pts_3d[:,2])
                    # self.stlPlot.draw()
                    ### convert the 3D model to 3D printer scale
                    # key_pts_3d = 1.1*160*key_pts_3d/0.16
                    minz = min(key_pts_3d[:,2])+0.0
                    key_pts_3d[:,2] -= minz
                    key_pts_3d[:,[0, 1]] = key_pts_3d[:,[1, 0]]
                    key_pts_3d[:,0] = -key_pts_3d[:,0]
                    ### save 3D tool-path ###
                    self.pathFile3D = '3d_toolpath.nc'
                    toolPathObj3D.contour2path(key_pts_3d,self.pathFile3D)
                    return key_pts_3d
        

    def ExtractContour(self,): ## Extract the contour and generate toolpath

        self.pathFile2D = '2d_toolpath.nc'
        # print (self.pathFile)
        # step = 0.004
        # feed = 0.005
        self.a.clear()
        ### Extract 2D contour and generate toolpath ###
        # scale_img2real = float(self.EntryBoxImgReal.get())
        # print (scale_img2real)
        # offsetX = float(self.EntryBoxOffsetX.get())
        # offsetY = float(self.EntryBoxOffsetY.get())
        scale_img2real = 1
        offsetX = 0
        offsetY = 0
        step = 0.5
        feed = 2
        if len(self.contour)>2:
            cnt_real_scale = self.contour
            cnt_center = np.mean(cnt_real_scale,axis=0)
            print('Contour center is in x=',cnt_center[0],'mm, y= ',cnt_center[1],' mm')
        else:
            print("No contour!")
            cnt_center = np.array([0,0])
            return

        # print(cnt_real_scale)
        z_path = len(cnt_real_scale)*[1]

        toolPathObj = toolpathGenerationFromContour(cnt_real_scale ,step,feed)
        # if self.rect is not None:
        #     z_path = toolPathObj.get_z_path(self.rect)
        self.key_pts = toolPathObj.scanHorizental()

        # offsetX = -min(self.key_pts[:,1])+self.needle_location[0][0]*real_scale_factor

        # self.path = np.transpose(np.array([self.key_pts[:,1]-320*real_scale_factor-offsetX,
        #     self.key_pts[:,0]-240*real_scale_factor-offsetY,len(self.key_pts)*[0]]))

        # self.path = np.transpose(np.array([self.key_pts[:,1]-offsetX,
        #     self.key_pts[:,0]-offsetY,len(self.key_pts)*[0]]))

        key_pts_2d = np.transpose(np.array([-cnt_real_scale[:,1],cnt_real_scale[:,0],z_path]))
        

        ### Align 2D and 3D tool-path ###
        path3d = self.ExtractContour3D()
        if path3d is not None:
            ### perform transformation: translational and rotational
            # path3d -= cnt_center
            final_pts = np.concatenate((key_pts_2d,path3d),axis=0)
            ### Plot the transformed 3D path
            self.a.plot(path3d[:,0],path3d[:,1],path3d[:,2])

        machine_scale_factor =1.1# 10/6.1/1.6
        cnt_machine_scale = key_pts_2d*machine_scale_factor
        # machine_scale_factor = 1/6.2

        toolPathObj.contour2path(cnt_machine_scale,self.pathFile2D)

        ### Plot 2D contour and 3D path ###  

        self.a.plot(key_pts_2d[:,0],key_pts_2d[:,1],key_pts_2d[:,2])
        self.a.scatter(0,0)

        self.a.set_xlabel('X direction /mm')
        self.a.set_ylabel('Y direction /mm')
        self.a.set_title('Toolpath planning')
        # self.a.set_aspect('equal', 'box')
        xmin = min(key_pts_2d[:,0])-20;xmax = max(key_pts_2d[:,0])+50;
        ymin = min(key_pts_2d[:,1])-10;ymax = max(key_pts_2d[:,1])+10
        self.a.set_xlim([xmin,xmax])
        self.a.set_ylim([ymin,ymax])
        # self.cntPlot.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self.cntPlot.draw()
        return

    def LoadPath(self):
        # import Tkconstants, tkFileDialog 
        # self.pathFile = filedialog.askopenfilename(initialdir = "C:\\Users\\houzh\\Google Drive\\PhD-Proposal\\software\\3D Printing Software",
        #     title = "Select file",
        #     filetypes = (("Gcode files","*.nc"),("Text files","*.txt"),("all files","*.*")))
        # print (self.pathFile)

        # Loading an existing stl file:
        from tkinter import filedialog
        self.stlpathFile = filedialog.askopenfilename(initialdir = "C:\\Users\\Administrator\\Google Drive\\PhD_defense\\Software_Related",
            title = "Select file",
            filetypes = (("Gcode files","*.stl"),("all files","*.*")))
        print (self.stlpathFile)
        # try:
        self.hand_mesh = mesh.Mesh.from_file(self.stlpathFile)
        self.b.cla()
        # self.b.scatter(0,0,0)
        self.b.set_xlabel('X direction')
        self.b.set_ylabel('Y direction')
        self.b.set_title('Display STL file') 
        # self.b.set_xlim([0, 0.5])
        # self.b.set_ylim([-0.3, 0])
        # self.b.set_zlim([-0.5, -0.4])
        # scale = self.hand_mesh.points.flatten(-1)
        # print(scale)
        # self.b.auto_scale_xyz(scale, scale, scale)
        # self.b.add_collection3d(art3d.Poly3DCollection(self.hand_mesh.vectors))
        # self.a.set_aspect('equal', 'box')
        # self.stlPlot.draw()

        # except Exception as e:
        #     print('Cannot find any STL file to show')
        #     return


    def PrintContour(self,): ## Direct print path by using the serial port communication

        f = open(self.pathFile,'r');
        verbose = True
        RX_BUFFER_SIZE = 2560
        settings_mode = False
        l_count = 0
        for line in f:
            # l_count += 1 # Iterate line counter
            # # l_block = re.sub('\s|\(.*?\)','',line).upper() # Strip comments/spaces/new line and capitalize
            l_block = line.strip()
            time_idx = l_block.find(';')
            delay_time = 0
            if time_idx>0:
                delay_time = l_block[time_idx+1:]
            self.s.write((l_block + '\n').encode()) # Send g-code block to tinyG
            time.sleep(float(delay_time))
        f.close()


    def StartFollow(self,):

        def follow_path():
            self.start_tracking = 1
            if self.start_tracking:
                threshold = 1
                path = self.path*1.1*0.16
                x = path[0,0]
                y = path[0,1]
                z = path[0,2]

                # if x**2+y**2+z**2>threshold:
                self.XYZA.GoToXYZ(x,y,z)
        Thread(target=follow_path).start()
        return
    def StopFollow(self,):
        self.start_tracking = 0

    ### Define button to control speed and flowrate manully
    def V_plus(self,speed):
        self.speed  = int(speed)+1;
        self.tSpeed.set(str(self.speed));
        # print(self.speed,self.tSpeed)
    def V_minus(self,speed):
        self.speed  = int(speed)-1;
        self.tSpeed.set(str(self.speed));
    def F_plus(self,flowrate):
        self.flowrate = int(flowrate)+1;
        self.flowRateV.set(str(self.flowrate));
    def F_minus(self,flowrate):
        self.flowrate = int(flowrate)-1;
        self.flowRateV.set(str(self.flowrate));
    def ClickAndCrop(self,event, x, y, flags, param):
        if self.cap1.isOpened():
            if event == cv2.EVENT_LBUTTONDOWN:
                self.cropPos = [(x,y)]
                self.cropping = True
            elif event == cv2.EVENT_LBUTTONUP:
                self.cropPos.append((x,y))
                self.cropping = False
                cv2.rectangle(self.rgb,self.cropPos[0],self.cropPos[1],(0,255,0),2)
                cv2.imshow('frame',self.rgb)

    def Connect(self,portNum):
        # Open grbl serial port
        self.s = serial.Serial(
            port = portNum,
            baudrate = 115200,
            parity = serial.PARITY_NONE,
            stopbits = serial.STOPBITS_ONE,
            bytesize = serial.EIGHTBITS)

        if(self.s):
            print ("Connection successful!")
            # Wake up grbl
            self.s.write(("\r\n\r\n").encode())
            time.sleep(2)   # Wait for grbl to initialize
            self.s.flushInput()  # Flush startup text in serial input
            l = 'G92 X0 Y0 Z0 A0\n'
            print ('Sending: ' + l,)
            self.s.write((l + '\n').encode()) # Send g-code block to grbl
            grbl_out = self.s.readline() # Wait for grbl response with carriage return
            print (' : ' + grbl_out.decode())
        else:
            print ("Failed to connect ...")
        self.speed = 'F100'
        self.XYZA.s =self.s
        self.XYZA.incXYZNum=self.incXYZ.get()
        self.XYZA.incANum=self.incA.get()
        self.XYZA.speed = self.speed
        return

    def Disconnect(self):

        ### Disconnect from COM3...
        if self.s is not None:
            self.s.close()

        ### Close cameras 
        self.closeCamSign = 0
        self.cap1.release()
        # self.cap2.release()
        cv2.destroyAllWindows()

        ### Stop threads
        if self.liveCamThread and self.liveCamThread.isAlive():
            self.liveCamThread._Thread_stop()
        if self.liveCamNeedleThread and self.liveCamNeedleThread.isAlive():
            self.liveCamNeedleThread._Thread_stop()

############################################ Monitor  #############################################
    def StartMonitoring(self):
        try:

            self.liveCamThread = Thread(target=self.LiveCamera,args=(self.camNum,self.cap1,self.c))
            self.connectThread = Thread(target=self.Connect,args=(self.portNum,))
            self.liveCamThread.start()
            self.connectThread.start()
        except:
            print ("Error: unable to show camera or connect to the board")
        return
    def PrintContourThread(self):
        try:
            # self.running.clear()
            # self.liveCamThread.stop()
            self.PrintThread = Thread(target=self.PrintContour)
            self.PrintThread.start()
        except:
            print ("Error: unable to show camera or connect to the board")
        return    

from tkinter import messagebox
def ask_quit():
    if messagebox.askokcancel("Quit", "You want to quit now? *sniff*"):
        root.destroy()   

if __name__ == "__main__":
    root = Tk()
    root.resizable()
    root.title('Bioprinting Controller')
    root.geometry('1600x1000')
    BioprinterControl(root)
    # root.protocol("WM_DELETE_WINDOW", ask_quit)
    root.mainloop()
