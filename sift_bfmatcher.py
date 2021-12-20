import cv2
import numpy as np
import requests, io, cv2
from PIL import Image

#https://bkshin.tistory.com/entry/OpenCV-29-%EC%98%AC%EB%B0%94%EB%A5%B8-%EB%A7%A4%EC%B9%AD%EC%A0%90-%EC%B0%BE%EA%B8%B0?category=1148027
#https://deep-learning-study.tistory.com/261
def start(img1, img2):
    response1 = requests.get(img1)
    response2 = requests.get(img2)
    img1 = np.array(Image.open(io.BytesIO(response1.content)))
    img2 = np.array(Image.open(io.BytesIO(response2.content)))
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    h = height1/height2
    w = width1/width2

    '''
    img2 = cv2.warpAffine(img2, np.float32([[float(h), 0, 0],
                        [0, float(w),0]]), (int(height2*h), int(width2*w), \
                            None, cv2.INTER_AREA))
    '''
    img2 = cv2.resize(img2, (int(width2 * w), int(height2 * h)), interpolation=cv2.INTER_CUBIC)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT 서술자 추출기 생성 ---①
    detector = cv2.xfeatures2d.SIFT_create()
    # 각 영상에 대해 키 포인트와 서술자 추출 ---②
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)

    # BFMatcher 생성, L1 거리, 상호 체크 ---③
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    # 매칭 계산 ---④
    matches = matcher.match(desc1, desc2)


    # 매칭 결과를 거리기준 오름차순으로 정렬 ---③
    matches = sorted(matches, key=lambda x:x.distance)
    # 최소 거리 값과 최대 거리 값 확보 ---④
    min_dist, max_dist = matches[0].distance, matches[-1].distance
    # 최소 거리의 15% 지점을 임계점으로 설정 ---⑤
    ratio = 0.05
    good_thresh = (max_dist - min_dist) * ratio + min_dist
    # 임계점 보다 작은 매칭점만 좋은 매칭점으로 분류 ---⑥
    good_matches = [m for m in matches if m.distance < good_thresh]
    print('matches:%d/%d, min:%.2f, max:%.2f, thresh:%.2f' \
            %(len(good_matches),len(matches), min_dist, max_dist, good_thresh))
    '''
    # 좋은 매칭 결과 선별
    good_matches = []

    for m in matches:
        if m.distance < 50:
            good_matches.append([m])

    print('# of kp1:', len(kp1))
    print('# of kp2:', len(kp2))
    print('# of matches:', len(matches))
    print('# of good_matches:', len(good_matches))
    '''
    # 좋은 매칭점의 queryIdx로 원본 영상의 좌표 구하기 ---③
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
    # 좋은 매칭점의 trainIdx로 대상 영상의 좌표 구하기 ---④
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])
    # 원근 변환 행렬 구하기 ---⑤
    mtrx, mask = cv2.findHomography(src_pts, dst_pts)
    # 원본 영상 크기로 변환 영역 좌표 생성 ---⑥
    h,w, = img1.shape[:2]
    pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
    # 원본 영상 좌표를 원근 변환  ---⑦
    dst = cv2.perspectiveTransform(pts,mtrx)
    # 변환 좌표 영역을 대상 영상에 그리기 ---⑧
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    # 매칭 결과 그리기 ---⑤
    res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # 결과 출력 
    cv2.imshow('BFMatcher + SIFT', res)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    pass