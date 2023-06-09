import numpy as np
import cv2
from matplotlib import pyplot as plt
import requests, io, cv2
from PIL import Image

#https://deep-learning-study.tistory.com/261

response1 = requests.get('https://bodydoctors.co.kr/data/goods/1/2019/04/37_tmp_2c4ce885ec6e40708ac22cf0821ad47d9246view.jpg')
response2 = requests.get('https://thumbnail8.coupangcdn.com/thumbnails/remote/492x492ex/image/retail/images/2019/12/02/18/2/944013b8-21c2-4455-9c6b-a3143a24cc2a.jpg')
img1 = np.array(Image.open(io.BytesIO(response1.content)))
img2 = np.array(Image.open(io.BytesIO(response2.content)))

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(500)

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=2)
plt.imshow(img3),plt.show()


if __name__ == '__main__':
    pass