import cv2
import numpy as np
import requests, io, cv2
from PIL import Image

#https://deep-learning-study.tistory.com/260

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

matcher = cv2.BFMatcher_create()
# 매칭 계산 ---④
matches = matcher.knnMatch(desc1, desc2, 2) # knnMatch로 특징점 2개 검출

# 좋은 매칭 결과 선별
good_matches = []
for m in matches: # matches는 두개의 리스트로 구성
    if m[0].distance / m[1].distance < 0.7: # 임계점 0.7
        good_matches.append(m[0]) # 저장

print('# of kp1:', len(kp1))
print('# of kp2:', len(kp2))
print('# of matches:', len(matches))
print('# of good_matches:', len(good_matches))

# 특징점 매칭 결과 영상 생성
dst = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)

cv2.namedWindow('dst',cv2.WINDOW_NORMAL)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()

if __name__ == '__main__':
    pass