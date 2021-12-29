import numpy as np
import cv2

img = cv2.imread("1.jpg")

poly = np.array([[10,300],[150,200],[300,300],[350,200],[310,40],[35,200]])
print(poly)
print(type(poly))
## (1) Crop the bounding rect
rect = cv2.boundingRect(poly)
x,y,w,h = rect
croped = img[y:y+h, x:x+w].copy()

## (2) make mask
poly = poly - poly.min(axis=0)

mask = np.zeros(croped.shape[:2], np.uint8)
cv2.drawContours(mask, [poly], -1, (255, 255, 255), -1, cv2.LINE_AA)

## (3) do bit-op
dst = cv2.bitwise_and(croped, croped, mask=mask)

## (4) add the white background
bg = np.ones_like(croped, np.uint8)*255
cv2.bitwise_not(bg,bg, mask=mask)
dst2 = bg+ dst

cv2.imshow('BFMatcher + SIFT', croped)

cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow('BFMatcher + SIFT', mask)

cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow('BFMatcher + SIFT', dst)

cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow('BFMatcher + SIFT', dst2)

cv2.waitKey()
cv2.destroyAllWindows()