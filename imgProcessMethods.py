### Image Process Methods ###
### Contour Extraction from Top View ###
### Volumetric Flowrate Calibration from side view ###

import numpy as np
import cv2
import cv2.aruco as aruco
import argparse
import glob
import math
from tkinter import *
from PIL import Image,ImageTk 
### customized methods ###
from learn_from_color import *
from learn_obj_color import *

class ImgProcessMethods():
    def __init__(self):
        ## load color classifier ###
        self.classifier = learn_from_color()
        print('Successfully load pre-trained SVM color classifier')
    def auto_canny(image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(np.uint8(image))
     
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
     
        # return the edged image
        return edged

    def ContourExtractArUcoMarker(self,img_rgb,show_button,needle_location):
        # mtx = np.array([[3.60485361e+03, 0, 1.64937012e+03],
        #                 [0.00000000e+00 ,3.59131010e+03, 1.93139068e+03],
        #                 [0.000000000, 0 ,1.00000000e+00]])

        # dist =  np.array([[ 2.13015194e-01 ,-1.36759896e+00 ,
        #                     -5.11614694e-04  ,3.63414300e-03,2.44071484e+00]]) 

        # m,n,c = img_rgb.shape
        # xfac = 320/n
        # yfac = 320/n

        # img_rgb_out = img_rgb.copy()
        # gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        # aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

        # ## detector parameters can be set here (list of detection parameters)
        # parameters = aruco.DetectorParameters_create()
        # # parameters.adaptiveThreshConstant = 50

        # corners,ids,rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

        # ## font for display the text
        # # font = cv2.FONT_HERSHEY_SIMPLEX
        # # if corners:
        # # print(ids)
        # ## check if the ids list is not empty

        # (thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # kernel1 = np.ones((3,3),np.uint8)
        # kernal2 = np.ones((3,3),np.uint8)
        # dilation = cv2.dilate(im_bw,kernel1,iterations = 1)
        # img_rgb_out = cv2.erode(dilation,kernal2,iterations = 1)       


        # if np.all(ids!=None):
        #     rvec,tvec,_ =aruco.estimatePoseSingleMarkers(corners,0.015,mtx,dist)
        #     (rvec-tvec).any()

        #     for i in range(rvec.shape[0]):
        #         aruco.drawAxis(img_rgb_out,mtx,dist,rvec[i,:,:],tvec[i,:,:],0.03)
        #         aruco.drawDetectedMarkers(img_rgb_out,corners)
        # print(corners.shape,corners)
        # ## get perspective transformation matrix
        # corners_on_img = [corners[0][0],corners[1][1],corners[2][2],corners[3][3]]
        # corners_on_img_cor = np.float([[0,0],[320,0],[0,240],[320,240]]) # 320x240
        # corners_on_printer = np.float([[0,0],[134,0],[0,78],[134,78]]) # 134x78 mm

        # M_img_cor = cv2.getPerspectiveTransform(corners_on_img,corners_on_img_cor)
        # M_img2printer = cv2.getPerspectiveTransform(corners_on_img,corners_on_printer)

        # img_rgb_out = cv2.wrapPerspective(img_rgb_out,M_img_cor)


        # ininital camera parameters
        mtx = np.array([[3.60485361e+03, 0, 1.64937012e+03],
        [0.00000000e+00 ,3.59131010e+03, 1.93139068e+03],
         [0.000000000, 0 ,1.00000000e+00]])

        dist =  np.array([[ 2.13015194e-01 ,-1.36759896e+00 ,-5.11614694e-04  ,3.63414300e-03,2.44071484e+00]]) 

        # video = 'http://155.246.5.89:8081'

        # cap = cv2.VideoCapture(0)
            
        ## termination criteria for the iterative algorithm 
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
        svm_clf = learn_obj_color()
        M_img_cor_old = []
        M_img_cor = []
        scale_fac = 1
        contour_out = []
        contour_out_real = []
        box = []
        center_transform = (1,1)
        frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        img_rgb_out = frame.copy()
        # ##show needle region ##
        # if len(needle_location)==1:
        #     center = (int(needle_location[0][0]),int(needle_location[0][1]))
        #     cv2.circle(img_rgb_out,center,10,(0,255,0),-1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

        ## detector parameters can be set here (list of detection parameters)
        parameters = aruco.DetectorParameters_create()
        # parameters.adaptiveThreshConstant = 50

        corners,ids,rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

        ## font for display the text
        font = cv2.FONT_HERSHEY_SIMPLEX

        # print(a.predict([[50,50,100]]))
        m,n,c = frame.shape
        xfac = 320/n
        yfac = 320/n
        Z = frame.reshape((-1,3))
        # print(Z)
        labels = svm_clf.predict(Z)
        centers = [[255,0,0],[0,255,0],[0,0,255]]

        label=(np.uint8(labels.flatten().reshape(m,n))) 

        cnts,hierarchy=cv2.findContours(label,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts)>0:
            largest_areas = sorted(cnts, key=cv2.contourArea)
            if len(largest_areas)>1:
                epsilon = 0.001 * cv2.arcLength(largest_areas[-1], True)
                # get minimal rectangle of it:
                # print(type(largest_areas[-1]),largest_areas[-1])
                # get approx polygons
                approx = cv2.approxPolyDP(largest_areas[-1], epsilon, True)
                # print approx.shape,type(approx)
                # draw approx polygons
                approx_cnt =  approx[:,0]
                # print approx_cnt.shape
                cnt_center = np.mean(approx_cnt,axis=0)
                # approx_cnt = np.append(approx_cnt,[cnt_center,[1,1],[640,1],[640,480],[1,480]],axis=0)
                approx_scale = np.transpose(np.array([(approx_cnt[:,0]-cnt_center[0])*scale_fac+cnt_center[0]
                ,(approx_cnt[:,1]-cnt_center[1])*scale_fac+cnt_center[1]]))
                # approx_scale = np.append(approx_scale,[[5,5],cnt_center,[640,320]],axis=0)
                # print approx_scale.shape,approx_scale
                cv2.drawContours(frame, [approx_scale.astype(int)], -1, (255, 0, 0), 3)
                contour_out =  approx_scale         

        # if corners:
        # print(ids)
        ## check if the ids list is not empty
        if np.all(ids!=None):
            rvec,tvec,_ =aruco.estimatePoseSingleMarkers(corners,0.05,mtx,dist)
            (rvec-tvec).any()
        
            for i in range(rvec.shape[0]):
                # aruco.drawAxis(frame,mtx,dist,rvec[i,:,:],tvec[i,:,:],0.03)
                aruco.drawDetectedMarkers(frame,corners)
                
            img_rgb_out = frame.copy()
            # print(ids[0],corners)
            if len(ids)==4 and show_button == 1:
                w = 402
                h = 234
                # w = 134
                # h = 78
                ##############################################################
                ##############      1(0,0)            3(w,0)

                
                ##############      0(0,h)            2(w,h)


                ##############      (0,2h)             (w,2h)
                ##############################################################
                mids = [3,0,2,1]
                # mids = [0,1,2,3]
                corner_defined = np.float32([[0,h],[0,0],[w,h],[w,0]]) # 320x240, 

                ## get perspective transformation matrix
                corners_on_img = np.float32([corners[0][0][mids[ids[0][0]]],corners[1][0][mids[ids[1][0]]],corners[2][0][mids[ids[2][0]]],corners[3][0][mids[ids[3][0]]]])
                corners_on_img_cor = np.float32([corner_defined[ids[0]],corner_defined[ids[1]],corner_defined[ids[2]],corner_defined[ids[3]]]) # 320x240
                corners_on_printer = np.float32([[0,0],[134,0],[0,78],[134,78]]) # 134x78 mm

                # print(corners_on_img[0:3],corners_on_img_cor[0:3])            
                M_img_cor = cv2.getPerspectiveTransform(corners_on_img,corners_on_img_cor)
                M_img2printer = cv2.getPerspectiveTransform(corners_on_img,corners_on_printer)

                # frame[label] = (255,0,0)
                # frame = cv2.bitwise_and(frame,frame,mask=label)
                # print(label)
                img_rgb_out = cv2.warpPerspective(frame,M_img_cor,(w,2*h))

                ##show transformed needle region ##
                if len(needle_location)==1:
                    center = np.array([needle_location[0][0],needle_location[0][1],1])
                    # needle_location_transform = np.matmul(M_img_cor,(center.T))
                    x = center[0]
                    y = center[1]
                    
                    M = M_img_cor
                    x_trans = (M[0][0]*x+M[0][1]*y+M[0][2])/(M[2][0]*x+M[2][1]*y+M[2][2])
                    y_trans = (M[1][0]*x+M[1][1]*y+M[1][2])/(M[2][0]*x+M[2][1]*y+M[2][2])
                    center_transform = (int(x_trans),int(y_trans))
                    cv2.circle(img_rgb_out,center_transform,10,(255,0,0),-1)

                    ## convert the contour to the real scale
                    if len(contour_out)>2:
                        M = M_img_cor
                        xs = contour_out[::2,0] # downsample the contour
                        ys = contour_out[::2,1]
                        
                        contour_out_real = np.array([
                            [((M[0][0]*xs[i]+M[0][1]*ys[i]+M[0][2])/(M[2][0]*xs[i]+M[2][1]*ys[i]+M[2][2])-x_trans),
                             ((M[1][0]*xs[i]+M[1][1]*ys[i]+M[1][2])/(M[2][0]*xs[i]+M[2][1]*ys[i]+M[2][2])-y_trans)]
                            for i in range(len(xs))])

                        ### get the outbounded rectangle
                        # contour_out_real_vec = np.array([[contour_out_real[i,:]] for i in range(len(contour_out_real))]) 
                        # cntt = np.array([[-0.65,-0.56],[-100.55,-0.54],[0.678,100.98],[-100.5,-100.8]])
                        # print(cntt)

                        fac = 0.3
                        cnt_center = np.mean(contour_out_real,axis=0)

                        # approx_cnt = np.append(approx_cnt,[cnt_center,[1,1],[640,1],[640,480],[1,480]],axis=0)
                        contour_out_real = np.transpose(np.array([(contour_out_real[:,0]-cnt_center[0])*fac+cnt_center[0]*fac
                        ,(contour_out_real[:,1]-cnt_center[1])*fac+cnt_center[1]*fac]))

                        rect = cv2.minAreaRect(contour_out_real.astype(int))
                        box = cv2.boxPoints(rect)
                        # print(max(contour_out_real[:,0])-min(contour_out_real[:,0]))
                        # print(max(contour_out_real[:,1])-min(contour_out_real[:,1]))
                        # print(contour_out_real)
                        # cv2.drawContours(img_rgb_out, [contour_out_real.astype(int)], -1, (0, 0, 255), 5)

                        # print(contour_out,contour_out_real)               
                # else:
                # if M_img_cor_old is not None:
                    # img_rgb_out = cv2.warpPerspective(frame,M_img_cor_old,(w,2*h))

            # if M_img_cor is not None:
            #     M_img_cor_old=M_img_cor


        ##show needle region ##
            else:
                if len(needle_location)==1:
                    center = (int(needle_location[0][0]),int(needle_location[0][1]))
                    cv2.circle(img_rgb_out,center,10,(0,255,0),-1)
            
        img_rgb_out_pil = cv2.resize(img_rgb_out,None,fx=xfac,fy=yfac)
        img_rgb_out = ImageTk.PhotoImage(image=Image.fromarray(img_rgb_out_pil))
        return img_rgb_out,center_transform,contour_out_real,box

    def ContourExtractSVM(self,img_rgb,count):
        dst = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2RGB)
        if(self.classifier):
            svm_clf = self.classifier
            m,n,c = img_rgb.shape
            xfac = 320/n
            yfac = 320/n
            
            Z = img_rgb.reshape((-1,3))
            # print(Z)
            labels = svm_clf.predict(Z)
            img_rgb_label=(np.uint8(labels.flatten().reshape(m,n)))
            (thresh, im_bw) = cv2.threshold(img_rgb_label, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            kernel1 = np.ones((3,3),np.uint8)
            kernal2 = np.ones((3,3),np.uint8)
            dilation = cv2.dilate(im_bw,kernel1,iterations = 1)
            dst = cv2.erode(dilation,kernal2,iterations = 1)

        img_rgb_out_pil = cv2.resize(dst,None,fx=xfac,fy=yfac)
        img_rgb_out = ImageTk.PhotoImage(image=Image.fromarray(img_rgb_out_pil))
        # if count%5==0:
            # cv2.imwrite('PrintingImages\\'+str(count)+'.jpg',dst)
        return img_rgb_out
    def ContourExtractKmean(self,img_rgb,K,scale_fac,kernal_num):
        # img_hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        xfac=scale_fac
        yfac=scale_fac

        m,n,c = img_rgb.shape
        xfac = 320/n
        yfac = 320/n

        img_hsv = img_rgb
        line_pixel_sum_set = []
        contour_out = []
        Z = img_hsv.reshape((-1,3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
        ret,label,center=cv2.kmeans(Z,K,None,criteria,5,cv2.KMEANS_RANDOM_CENTERS)
         # Now convert back into uint8, and make original image
        frameWithCnt = cv2.resize(img_rgb,None,fx=xfac,fy=yfac)
        edge_hsv_out = ImageTk.PhotoImage(image=Image.fromarray(frameWithCnt))
        kmeanLines_small_tk_out = edge_hsv_out
        if len(center)>1:
            center = np.uint8(center)
            res = center[label.flatten()]

            kmeanMap = res.reshape((img_hsv.shape))
            kmeanLines = kmeanMap.copy()
            
            ## Find lines in hsv image
            gray = cv2.cvtColor(kmeanLines,cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            # edges = self.auto_canny(blurred)
            edges = cv2.Canny(blurred,10,200)
            labelimg = np.uint8(label.flatten().reshape(edges.shape))
            minLineLength = 2
            maxLineGap = 50
            lines = cv2.HoughLines(edges,1,np.pi/180,160,None,0,0)
            final_inter_x,final_inter_y = 0,0
            final_pts = (final_inter_x,final_inter_y)
            # if lines is not None:
            #     for i in range(len(lines)):
            #         rho = lines[i][0][0]
            #         theta = lines[i][0][1]
            #         if theta < np.pi/1.9 and theta >np.pi/2.1 or theta<np.pi/10 and theta>-np.pi/10:
            #         # if theta:
            #             a = math.cos(theta)
            #             b = math.sin(theta)
            #             x0 = a * rho
            #             y0 = b * rho
            #             pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            #             pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                    
            #             # print pt1,pt2
            #             cv2.line(kmeanLines, pt1, pt2, (255,0,0), 3)
            #             if math.fabs(pt1[0]-pt2[0]) > math.fabs(pt1[1]-pt2[1]):## horizontal line
            #                 final_inter_y = int((pt1[1]+pt2[1])/2)
            #             if math.fabs(pt1[1]-pt2[1]) > math.fabs(pt1[0]-pt2[0]):## vertical line
            #                 final_inter_x = int((pt1[0]+pt2[0])/2)
            #             final_pts = (final_inter_x,final_inter_y)
            #             if final_inter_x and final_inter_y:
            #                 cv2.circle(kmeanLines, final_pts,10, (255,0,0), 3)

                
            #                 pt_x = final_pts[0]
            #                 pt_y = final_pts[1]
            #                 # print pt_x,pt_y
            #                 stop_sign = 1
            #                 width = 10
            #                 while stop_sign and pt_y>240:
            #                     line_pixel_sum = 0
            #                     for xidx in range(pt_x-width,pt_x+width):
            #                         if labelimg[xidx,pt_y] != labelimg[xidx+1,pt_y]:
            #                             line_pixel_sum+=1
            #                     # line_pixel_sum = sum(labelimg[pt_x-width:pt_x+width,pt_y]) 
            #                     pt_y = pt_y-1
            #                     # print pt_y
            #                     line_pixel_sum_set.append(line_pixel_sum)
            #                     # if line_pixel_sum_set[-1] < line_pixel_sum:
            #                     #     stop_sign = 0
                            

            
            kmeanLines_small = cv2.resize(kmeanLines,None,fx=xfac,fy=yfac)
            kmeanLines_small_tk = ImageTk.PhotoImage(image=Image.fromarray(kmeanLines_small))

            ## Find contour of hsv image
            ret,grayedge = cv2.threshold(cv2.cvtColor(kmeanMap, cv2.COLOR_RGB2GRAY), 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            kernel1 = np.ones((10,10),np.uint8)
            kernal2 = np.ones((kernal_num,kernal_num),np.uint8)
            dilation = cv2.dilate(grayedge,kernel1,iterations = 1)
            dst = cv2.erode(dilation,kernal2,iterations = 1)
            # img,cnts,hier = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts,hier = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(cnts)>0:
                largest_areas = sorted(cnts, key=cv2.contourArea)
                # print len(largest_areas)
                # for i in range(len(largest_areas)):
                if len(largest_areas)>1:
                    # print i
                    epsilon = 0.001 * cv2.arcLength(largest_areas[-1], True)
                    # get approx polygons
                    approx = cv2.approxPolyDP(largest_areas[-1], epsilon, True)
                    # print approx.shape,type(approx)
                    # draw approx polygons
                    
                    approx_cnt =  approx[:,0]
                    # print approx_cnt.shape
                    cnt_center = np.mean(approx_cnt,axis=0)
                    # approx_cnt = np.append(approx_cnt,[cnt_center,[1,1],[640,1],[640,480],[1,480]],axis=0)
                    approx_scale = np.transpose(np.array([(approx_cnt[:,0]-cnt_center[0])*scale_fac+cnt_center[0]
                    ,(approx_cnt[:,1]-cnt_center[1])*scale_fac+cnt_center[1]]))
                    # approx_scale = np.append(approx_scale,[[5,5],cnt_center,[640,320]],axis=0)
                    # print approx_scale.shape,approx_scale
                    cv2.drawContours(img_rgb, [approx_scale.astype(int)], -1, (0, 0, 255), 3)
                    contour_out =  approx_scale
                frameWithCnt = cv2.resize(img_rgb,None,fx=xfac,fy=yfac)
                edge_hsv_out = ImageTk.PhotoImage(image=Image.fromarray(frameWithCnt))
                kmeanLines_small_tk_out = kmeanLines_small_tk
        else:
            frameWithCnt = cv2.resize(img_rgb,None,fx=xfac,fy=yfac)
            edge_hsv_out = ImageTk.PhotoImage(image=Image.fromarray(frameWithCnt))
            kmeanLines_small_tk_out = kmeanLines_small_tk       
        return img_rgb,contour_out,edge_hsv_out,kmeanLines_small_tk_out,line_pixel_sum_set


    def FindNeedleAndPlatform(self,img_rgb,K):
        img_hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        Z = img_hsv.reshape((-1,3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
        ret,label,center=cv2.kmeans(Z,K,None,criteria,5,cv2.KMEANS_RANDOM_CENTERS)

        if len(center)>1:
            center = np.uint8(center)
            res = center[label.flatten()]
            kmeanMap = res.reshape((img_hsv.shape))

        gray = cv2.cvtColor(kmeanMap,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150,apertureSize = 3)

        minLineLength = 100
        maxLineGap = 10
        lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(kmeanMap,(x1,y1),(x2,y2),(0,255,0),2)

        kmeanMap_small = cv2.resize(kmeanMap,None,fx=0.5,fy=0.5)
        kmeanMap_small_tk = ImageTk.PhotoImage(image=Image.fromarray(kmeanMap_small))
        return kmeanMap_small_tk
