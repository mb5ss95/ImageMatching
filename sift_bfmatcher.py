import cv2
import numpy as np
import requests, io, cv2
from PIL import Image

#https://deep-learning-study.tistory.com/261

response1 = requests.get('https://bodydoctors.co.kr/data/goods/1/2019/04/37_tmp_2c4ce885ec6e40708ac22cf0821ad47d9246view.jpg')
response2 = requests.get('http://img4.tmon.kr/cdn3/deals/2020/05/08/2277913066/2277913066_front_ea7a587bb1.jpg')
img1 = np.array(Image.open(io.BytesIO(response1.content)))
img2 = np.array(Image.open(io.BytesIO(response2.content)))

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# SIFT 서술자 추출기 생성 ---①
detector = cv2.xfeatures2d.SIFT_create(50000)
# 각 영상에 대해 키 포인트와 서술자 추출 ---②
kp1, desc1 = detector.detectAndCompute(gray1, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)

# BFMatcher 생성, L1 거리, 상호 체크 ---③
matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
# 매칭 계산 ---④
matches = matcher.match(desc1, desc2)
# 매칭 결과 그리기 ---⑤
res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
                      flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# 결과 출력 
cv2.imshow('BFMatcher + SIFT', res)
cv2.waitKey()
cv2.destroyAllWindows()

if __name__ == '__main__':
    pass