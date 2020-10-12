import numpy as np
import argparse
import imutils
import cv2
'''
ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",help=" ")
arg=vars(ap.parse_args())

image=cv2.imread(args["image"])
'''
def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# 从右往左或者从下往上时，reverse设置为True,
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# 如果从上到下或者从下到上排序时，i=1
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	#此处的boudingBoxes推测是一个二维的结构。由于boudingRect函数会返回
	#一个列表，包含四个变量，x,y,w,h。
	#x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
	
	#zip可以将对象中对应元素打包成一个个元组
	#假设a是被列表,zip(*a)就是将a解压，返回二维矩阵式

	#sorted是排序函数 sorted(iterable, key=None, reverse=False)
	#literable 为可迭代对象
        #reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
	#key=lambda b:b[1][i]此项是按照b[1][i]进行排序，
	#b[1][0]就是x, b[1][1]就是y。
	
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)
def draw_contour(image, c, i):
	# compute the center of the contour area and draw a circle
	# representing the center
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	# draw the countour number on the image
	cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (255, 255, 255), 2)
	# return the image with the contour number drawn on it
	return image

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
ap.add_argument("-m", "--method", required=True, help="Sorting method")
args = vars(ap.parse_args())
# load the image and initialize the accumulated edge image
image = cv2.imread(args["image"])
accumEdged = np.zeros(image.shape[:2], dtype="uint8")
# loop over the blue, green, and red channels, respectively
for chan in cv2.split(image):
	# blur the channel, extract edges from it, and accumulate the set
	# of edges for the image
	chan = cv2.medianBlur(chan, 11)
	edged = cv2.Canny(chan, 50, 200)
	accumEdged = cv2.bitwise_or(accumEdged, edged)
# show the accumulated edge map
cv2.imshow("Edge Map", accumEdged)

# find contours in the accumulated image, keeping only the largest
# ones
cnts = cv2.findContours(accumEdged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
orig = image.copy()
# loop over the (unsorted) contours and draw them
for (i, c) in enumerate(cnts):
	orig = draw_contour(orig, c, i)			#上面的函数
# show the original, unsorted contour image
cv2.imshow("Unsorted", orig)
# sort the contours according to the provided method
(cnts, boundingBoxes) = sort_contours(cnts, method=args["method"]) #调用上面的函数
# loop over the (now sorted) contours and draw them
for (i, c) in enumerate(cnts):
	draw_contour(image, c, i)
# show the output image
cv2.imshow("Sorted", image)
cv2.waitKey(0)