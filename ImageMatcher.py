import cv2
import numpy as np
import requests, io
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim

class ImageMatcher():

    def __init__(self):
        pass
        
    # 요청한 사진을 RGB 배열로 반환해줌
    def request_img(self, url):
        return np.array(Image.open(io.BytesIO(requests.get(url).content)))
    
    def img_to_np(self, path):
        return np.array(Image.open(path))

    # 이미지 조절
    # 큰 이미지를 축소하여, 작은 이미지 사이즈에 맞는 Width, Height를 반환함
    def compare_size(self, img1, img2):
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]

        height =  height1 if height1 < height2 else height2
        width = width1 if width1 < width2 else width2

        return width, height
    
    def SIFT(self, gray1, gray2):
        # SIFT 서술자 추출기 생성 ---①
        detector = cv2.SIFT_create()
        # 각 영상에 대해 키 포인트와 서술자 추출 ---②
        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(gray2, None)

        return kp1, desc1, kp2, desc2
    
    def SURF(self, gray1, gray2):
        # SURF 서술자 추출기 생성 ---①
        detector = cv2.xfeatures2d.SURF_create()

        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(gray2, None)

        return kp1, desc1, kp2, desc2
    
    def ORB(self, gray1, gray2):
        # ORB 서술자 추출기 생성 ---①
        detector = cv2.ORB_create()
        # 각 영상에 대해 키 포인트와 서술자 추출 ---②
        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(gray2, None)

        return kp1, desc1, kp2, desc2
    
    def FLANN(self, desc1, desc2, ratio=0.15):
        # 인덱스 파라미터와 검색 파라미터 설정 ---①
        # sift, surf를 사용할 때
        # FLANN_INDEDX_KDTREE = 1
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

        # orb를 사용할 때
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)

        # Flann 매처 생성 ---③
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        # 매칭 계산 ---④
        matches = matcher.match(desc1, desc2)

        # 매칭 결과를 거리기준 오름차순으로 정렬 ---③
        matches = sorted(matches, key=lambda x:x.distance)

        # 최소 거리 값과 최대 거리 값 확보 ---④
        min_dist, max_dist = matches[0].distance, matches[-1].distance

        good_thresh = (max_dist - min_dist) * ratio + min_dist

        # 임계점 보다 작은 매칭점만 좋은 매칭점으로 분류 ---⑥
        good_matches = [m for m in matches if m.distance < good_thresh]

        return good_matches

    def BFMatcher(self, desc1, desc2, ratio=0.15): # 최소 거리의 15% 지점을 임계점으로 설정 ---⑤
        '''
        matches[0]에는 queryIdx, trainIdx, distance 정보가 포함되는데 각각의 의미는 아래와 같다.
        (1) queryIdx
        : 기준이 되는 descriptor 및 keypoint의 index이다.
            matches[0]는 desA[0]를 기준으로 삼기 때문에
            matches[0].queryIdx = 0 이다.
        (2) trainIdx
        : desA[0]과 매칭된 image B, des의 index에 해당한다.
        (3) distance
        : desA[0]와 매칭된 desB의 des 사이의 거리( = 유사도 )값이다.

        1. cv2.NORM_L1
        2. cv2.NORM_L2
        3. cv2.NORM_L2SQR
        4. cv2.NORM_HAMMING
        5. cv2.NORM_HAMMING2
        세 가지 유클리드 거리 측정법과 두 가지 해밍 거리 측정법 중에 선택을 할 수 있습니다. 
        SIFT와 SURF 디스크립터 검출기의 경우 NORM_L1, NORM_L2가 적합하고 ORB로 디스크립터 검출기의 경우 NORM_HAMMING이 적합함
        '''
        # BFMatcher 생성, L1 거리, 상호 체크 ---③
        matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        # 매칭 계산 ---④
        matches = matcher.match(desc1, desc2)

        # 매칭 결과를 거리기준 오름차순으로 정렬 ---③
        matches = sorted(matches, key=lambda x:x.distance)

        # 최소 거리 값과 최대 거리 값 확보 ---④
        min_dist, max_dist = matches[0].distance, matches[-1].distance

        good_thresh = (max_dist - min_dist) * ratio + min_dist

        # 임계점 보다 작은 매칭점만 좋은 매칭점으로 분류 ---⑥
        good_matches = [m for m in matches if m.distance < good_thresh]

        # 좋은 매칭점의 queryIdx로 원본 영상의 좌표 구하기 ---③
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
        # 좋은 매칭점의 trainIdx로 대상 영상의 좌표 구하기 ---④
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])

        return src_pts, dst_pts, good_matches
        # res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        # cv2.imshow('BFMatcher + SIFT', res)

    def compare_ssim_resize(self, img1, img2, src_pts, dst_pts):
        # 최외각 다각형 구하기
        print(src_pts)
        print(dst_pts)
        src_maxY, src_maxX = max(src_pts)
        src_minY, src_minX = min(src_pts)
        src_maxY, src_maxX = max(dst_pts)
        src_minY, src_minX = min(dst_pts)
        rect1 = cv2.boundingRect(src_pts)
        rect2 = cv2.boundingRect(dst_pts) # returns (x,y,w,h) of the rect
        # img1 = img1[rect1[1]: rect1[1] + rect1[3], rect1[0]: rect1[0] + rect1[2]]
        # img2 = img2[rect2[1]: rect2[1] + rect2[3], rect2[0]: rect2[0] + rect2[2]]

        # height1, width1 = img1.shape[:2]
        # height2, width2 = img2.shape[:2]

        # width, height = self.compare_size(img1, img2)
        
        # img1 = cv2.resize(img1, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        # img2 = cv2.resize(img2, (int(width), int(height)), interpolation=cv2.INTER_AREA)

        # print("img1 resize : ", height,",", width)
        # print("img2 resize : ", height,",", width)

        # grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # res = cv2.drawMatches(img1, [], img2, [], [], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        # cv2.imshow('SSIM + resize', res)

        # score, diff = compare_ssim(grayA, grayB, full=True)
        # print(f'SSIM2 : {score:.6f}')

        # return width1, height1, width2, height2
    
if __name__ == "__main__":
    imageMatcher = ImageMatcher()

    # url로 이미지 읽어 올거면
    # url1 = "https://www.shinailbo.co.kr/news/photo/202110/1466366_661801_2145.jpg"
    # url2 = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F36W98%2FbtqDVV2stjO%2FOxObRb2T25g0oiZ9njZPHK%2Fimg.jpg"
    # img1 = imageMatcher.request_img(url1)
    # img2 = imageMatcher.request_img(url2)

    # 로컬에서 이미지 읽어 올거면
    url1 = "./images/banana1.jpg"
    url2 = "./images/banana2.jpg"
    img1 = imageMatcher.img_to_np(url1)
    img2 = imageMatcher.img_to_np(url2)

    width, height = imageMatcher.compare_size(img1, img2)

    resizeImg1 = cv2.resize(img1, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    resizeImg2 = cv2.resize(img2, (int(width), int(height)), interpolation=cv2.INTER_AREA)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, desc1, kp2, desc2 = imageMatcher.SIFT(gray1, gray2)

    # [[h1, w1], [h2, w2], [h3, w3],,,,,[hn, wn]]
    src_pts, dst_pts, good_matches = imageMatcher.BFMatcher(desc1, desc2, 0.50)
    height1, width1 = resizeImg1.shape[:2]
    height2, width2 = resizeImg2.shape[:2]
    
    print(f"heigth, width {height} {width}")
    print(f"resizeImg1 {height1} {width1}")
    print(f"resizeImg2 {height2} {width2}")

    src_maxY = max(src_pts, key=lambda item: item[0])[0]  # 최대 y 값
    src_maxX = max(src_pts, key=lambda item: item[1])[1]  # 최대 x 값
    src_minY = min(src_pts, key=lambda item: item[0])[0]  # 최소 y 값
    src_minX = min(src_pts, key=lambda item: item[1])[1]  # 최소 x 값

    dst_maxY = max(dst_pts, key=lambda item: item[0])[0]  # 최대 y 값
    dst_maxX = max(dst_pts, key=lambda item: item[1])[1]  # 최대 x 값
    dst_minY = min(dst_pts, key=lambda item: item[0])[0]  # 최소 y 값
    dst_minX = min(dst_pts, key=lambda item: item[1])[1]  # 최소 x 값

    rect1 = cv2.boundingRect(src_pts)
    rect2 = cv2.boundingRect(dst_pts)

    img1 = img1[rect1[1]: rect1[1] + rect1[3], rect1[0]: rect1[0] + rect1[2]]
    img2 = img2[rect2[1]: rect2[1] + rect2[3], rect2[0]: rect2[0] + rect2[2]]
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    '''
    h, w 383 198
    h, w 358 623
    '''
    print(f"h, w {height1} {width1}")
    print(f"h, w {height2} {width2}")

    w, h = imageMatcher.compare_size(img1, img2)
    
    print(f"h, w {h} {w}")
    who = -1
    if (height1 + width1) < (height1 + width1):
        who = 0
    else:
        who = 1

    # 여기서 작은 크기를 가지고, 상대 이미지에서 찾아가면서 가장 많은 포인트를 가지고 있는 이미지에서 maxY, maxX, minY, minX를 구함
    # 그리고 거기서 비교 가자
    
    res = cv2.drawMatches(img1, [], img2, [], [], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('변경후', res)

    res = cv2.drawMatches(resizeImg1, kp1, resizeImg2, kp2, good_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('BFMatcher + SIFT', res)
    cv2.waitKey()
    cv2.destroyAllWindows()

    '''
    1. 비교할 이미지 2개를 가져온다.
    2. sift + bfmatcher, surf + bfmatcher, orb + flann 중에서 선택한다.
    3. 특징점을 추출한다.
    4. 특징점을 토대로 이미지를 슬라이싱 한다.
    5. 보여준다.
    '''