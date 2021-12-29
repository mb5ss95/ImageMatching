import numpy as np
import cv2

img = cv2.imread("1.jpg")

height = img.shape[0]
width = img.shape[1]

mask = np.zeros((height, width), dtype=np.uint8)
points = np.array([[[10,150],[150,100],[300,150],[350,100],[310,20],[35,10]]])
cv2.fillPoly(mask, points, (255))

res = cv2.bitwise_not(img,img,mask = mask)

rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

cv2.imshow("cropped" , cropped )
cv2.imshow("same size" , res)
cv2.waitKey(0)