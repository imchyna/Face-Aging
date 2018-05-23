# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
from glob import *

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=128)  # 128 is your data size 

# load the input image, resize it, and convert it to grayscale
imgpath = '../DataSet/CACD-crop-test/*.jpg'
imgfiles = glob(imgpath)
savepath = '../DataSet/CACD-align-test'

for _, i_img in enumerate(imgfiles):
    image = cv2.imread(i_img)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for rect in rects:
       faceAligned = fa.align(image, gray, rect)
       savename = savepath+'/'+str(i_img).split('/')[3]
       cv2.imwrite(savename,faceAligned)
