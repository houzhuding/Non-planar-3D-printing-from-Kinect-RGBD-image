import cv2
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import svm
import re 
import imutils

# ix,iy = -1,-1
# # mouse callback function
# def draw_circle(event,x,y,flags,param):
#     global ix,iy
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         # cv2.circle(img,(x,y),100,(255,0,0),-1)
#         ix,iy = x,y
#         print(img[y,x])
# cv2.namedWindow('image')        
# cv2.setMouseCallback('image',draw_circle)
# img = cv2.imread('3.bmp')
# hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# while(1):
#     # cv2.imshow('image',cv2.hconcat([img,hsv]))
#     cv2.imshow('image',hsv)
#     k = cv2.waitKey(20) & 0xFF
#     if k == 27:
#         break
#     # elif k == ord('a'):
#     #     print ix,iy
# cv2.destroyAllWindows()

def learn_obj_color():
    ### Load background color ###
    with open('bg.txt') as bg:
        line = bg.read()
    color_set = []
    # print(line.split('\n'))
    len_bg = 0
    for pixel in line.split('\n'):
        colors = re.findall(r'\d+',pixel)
        color = [int(s) for s in colors]
        if color:
            color_set.append(color)
            len_bg +=1
            # print(color)
    color_mat_bg = np.array(color_set)
    color_label_bg = np.zeros(len_bg)

    ### Load hydrogel color ###
    with open('obj.txt') as gel:
        line = gel.read()
    color_set = []
    # print(line.split('\n'))
    len_gel = 0
    for pixel in line.split('\n'):
        colors = re.findall(r'\d+',pixel)
        color = [int(s) for s in colors]
        if color:
            color_set.append(color)
            len_gel+=1
            # print(color)
    color_mat_gel = np.array(color_set)
    color_label_gel = np.zeros(len_gel)+255

    # ### Load needle color ###
    # with open('PrintingImages\\needle.txt') as needle:
    #     line = needle.read()
    # color_set = []
    # # print(line.split('\n'))
    # len_needle = 0
    # for pixel in line.split('\n'):
    #     colors = re.findall(r'\d+',pixel)
    #     color = [int(s) for s in colors]
    #     if color:
    #         color_set.append(color)
    #         len_needle+=1
    #         # print(color)
    # color_mat_needle = np.array(color_set)
    # color_label_needle = np.zeros(len_needle)+128
    # # print(color_label_needle)


    clf = svm.SVC(gamma = 'scale')
    X = np.concatenate((color_mat_bg,color_mat_gel),axis=0)
    y = np.concatenate((color_label_bg,color_label_gel))
    clf.fit(X,y)
    # dec = clf.decision_function([[1]])
    svp = np.array((clf.support_vectors_))
    # print(svp)

    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection='3d')
    # ax.scatter(color_mat_bg[:,0],color_mat_bg[:,1],color_mat_bg[:,2])
    # ax.scatter(color_mat_gel[:,0],color_mat_gel[:,1],color_mat_gel[:,2])
    # ax.scatter(color_mat_needle[:,0],color_mat_needle[:,1],color_mat_needle[:,2])
    # ax.scatter(svp[:,0],svp[:,1],svp[:,2])
    # ax.legend(['Bg','Gel','Needle','Support Vectors'])
    # plt.show()
    return clf

# svm_clf = learn_obj_color()
# # print(a.predict([[50,50,100]]))
# test_img = cv2.imread('1.bmp')
# # test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2HSV)
# m,n,c = test_img.shape
# xfac = 320/n
# yfac = 320/n
# img_rgb_out = test_img.copy()
# Z = test_img.reshape((-1,3))
# # print(Z)
# labels = svm_clf.predict(Z)
# centers = [[255,0,0],[0,255,0],[0,0,255]]

# label=(np.uint8(labels.flatten().reshape(m,n)))
# print(label)
# # for x in range(m):
# #     for y in range(n):
# #         rgb_val = test_img[x,y]
# #         if svm_clf.predict([rgb_val])==[0]:
# #             img_rgb_out[x,y] = (255,0,0)
# #         elif svm_clf.predict([rgb_val])==[1]:
# #             img_rgb_out[x,y] = (0,255,0)
# #         else:
# #             img_rgb_out[x,y] = (0,0,255)
# cv2.imshow('out',label)
# cv2.waitKey(0)