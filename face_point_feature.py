import os
import numpy as np 
import pandas as pd 
pd.options.display.float_format = '{:.2f}'.format
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import cv2
import random
import dlib
import imutils
from imutils import face_utils






# load the pre-trained 68 landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/content/drive/MyDrive/face_mask/shape_predictor_68_face_landmarks.dat')

# loop through images in directory
for file in sorted(os.listdir("/content/drive/MyDrive/face_mask")):
  if file.startswith(str(0)):
    image_path = "/content/drive/MyDrive/face_mask/" + file

    image = cv2.imread(image_path)
    image = imutils.resize(image, width=500)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # if landmarks could apply to the image, crop the image based on nose/ mouth area,
    # if not print out the file name
    if detector(image):
      rect = detector(image)[0]
      sp = predictor(image, rect)
      landmarks = np.array([[p.x, p.y] for p in sp.parts()])
      outline = landmarks[[*range(67,29,-1)]]
      outline_nose = landmarks[[*range(36,29,-1)]]
      outline_mouth = landmarks[[*range(67,49,-1)]]

      # get the nose and mouth area
      crop_nose = image[min(outline_nose[:,1]):max(outline_nose[:,1]), min(outline_nose[:,0]):max(outline[:,0])]
      crop_mouth = image[min(outline_mouth[:,1]):max(outline_mouth[:,1]), min(outline_mouth[:,0]):max(outline[:,0])]
      crop_nose_mouth = image[min(outline[:,1]):max(outline[:,1]), min(outline[:,0]):max(outline[:,0])]
      
      crop_nose = cv2.resize(crop_nose, (500,500))
      crop_mouth = cv2.resize(crop_mouth, (500,500))
      crop_nose_mouth = cv2.resize(crop_nose_mouth, (500,500))

      # write 3 files to directory, only nose, only mouth, nose and mouth 
      cv2.imwrite("nose" + file, crop_nose)
      cv2.imwrite("mouth" + file, crop_mouth)
      cv2.imwrite("nose_mouth" + file, crop_nose_mouth)
      
      plt.figure(figsize=(10,10))
      plt.imshow(crop_nose)
      plt.xticks([])
      plt.yticks([])
      plt.show()
    else:
      print("cannot detect image",file)