import numpy as np
import argparse
import imutils
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",help="path to the image file")
args=vars(ap.parse_args())

image=cv2.imread(args["image"])
lower=np.array([0,0,0])
upper=np.array([15,15,15])
shapeMask=cv2.inRange(image,lower,upper) #颜色低于lower,或者高于upper的，
cnts=cv2.findContours(shapeMask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)
print("I found {} black shapes".format(len(cnts)))
cv2.imshow("Mask",shapeMask)
cv2.imwrite("find_shapes.png",shapeMask)
for c in cnts:
    cv2.drawContours(image,[c],-1,(0,0,255),2)
    cv2.imshow("image",image)
    cv2.waitKey(0)