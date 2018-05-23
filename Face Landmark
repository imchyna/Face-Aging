from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from glob import *
import math


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# load the input image, resize it, and convert it to gray''scale
imgpath = '../DataSet/UTKFace-train/*.jpg'  #
savepath1 = '../DataSet/UTKFace-1/'
savepath2 = '../DataSet/UTKFace-0/'
# imgpath = '../DataSet/test/*.jpg'
img_files = glob(imgpath)

for _,i_img in enumerate(img_files):
    image0 = cv2.imread(i_img)
    image = image0
    
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
       
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # calculate the distances between the eyes outter corners and cheek dis0
        l_dis0 = math.sqrt((shape[31, 0] - shape[33, 0]) ** 2 + (shape[31, 1] - shape[33, 1]) ** 2)  
        r_dis0 = math.sqrt((shape[35, 0] - shape[33, 0]) ** 2 + (shape[35, 1] - shape[33, 1]) ** 2)
        # calculate the distances between the nose corners and nose center dis1
        l_dis1 = math.sqrt((shape[36, 0] - shape[0, 0]) ** 2 + (shape[36, 1] - shape[0, 1]) ** 2)
        r_dis1 = math.sqrt((shape[45, 0] - shape[16, 0]) ** 2 + (shape[45, 1] - shape[16, 1]) ** 2)

        dis0 = math.fabs(l_dis0 - r_dis0)
        dis1 = math.fabs(l_dis1 - r_dis1)
        print 'dis0:  ', dis0
        print 'dis1:  ', dis1
        # dis0 < 6 and dis1 <60 forward face for CACD
        if dis1>80:  # if is not face-forward  for UTKFace   
            savename = savepath2 + i_img.split('/')[3]    
        else:
            savename = savepath1 + i_img.split('/')[3]
        cv2.imwrite(savename, image0)
        
        # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow("Output", image)
            cv2.waitKey(20)
            
        cv2.destroyAllWindows()
    
