import argparse
import imutils
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to the input image")
args=vars(ap.parse_args())

image =cv2.imread(args["image"])
gray =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  #转为灰度图
blurred=cv2.GaussianBlur(gray,(5,5),0)      # 高斯模糊，(5,5)是内核大小，
thresh=cv2.threshold(blurred,200,225,cv2.THRESH_BINARY_INV)[1]  #二值化处理，最后一个参数为处理方式，本条命令是色值高于60的设置为255
cv2.imshow("",thresh)
cv2.waitKey(0)
#findContours 接受的图像参数必须为二值图，所以上面先讲将图像转化灰度图，然后再进行二值化处理
cnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)  #返回cnts中的countors(轮廓)

# loop over the contours
i=0
for c in cnts:
	# compute the center of the contour
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	# draw the contour and center of the shape on the image
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
	cv2.putText(image, "center", (cX - 20, cY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	# show the image
	cv2.imshow("Image", image)
	cv2.waitKey(0)

