import cv2
import numpy as np
from matplotlib import pyplot as plt
import requests, io, cv2
from PIL import Image


response1 = requests.get('https://bodydoctors.co.kr/data/goods/1/2019/04/37_tmp_2c4ce885ec6e40708ac22cf0821ad47d9246view.jpg')
response2 = requests.get('https://thumbnail8.coupangcdn.com/thumbnails/remote/492x492ex/image/retail/images/2019/12/02/18/2/944013b8-21c2-4455-9c6b-a3143a24cc2a.jpg')
img1 = np.array(Image.open(io.BytesIO(response1.content)))
img2 = np.array(Image.open(io.BytesIO(response2.content)))

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# SIFT 서술자 추출기 생성 ---①
detector = cv2.xfeatures2d.SIFT_create(5000)

# 각 영상에 대해 키 포인트와 서술자 추출 ---②
kp1, des1 = detector.detectAndCompute(gray1, None)
kp2, des2 = detector.detectAndCompute(gray2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

good_matches = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.75*n.distance:
        matchesMask[i]=[1,0]
        good_matches.append(1)
    else:
        good_matches.append(0)

print('# of kp1:', len(kp1))
print('# of kp2:', len(kp2))
print('# of matches:', len(matches))
print('# of good_matches:', good_matches.count(1)/len(good_matches))

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

# 결과 출력 
cv2.imshow('BFMatcher + SIFT', img3)
cv2.waitKey()
cv2.destroyAllWindows()

if __name__ == '__main__':
    pass