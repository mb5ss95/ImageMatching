import cv2
import numpy as np
import requests, io, cv2
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim

def compare_sift_bfmatch(img1, img2, gray1, gray2):
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

    res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('BFMatcher + SIFT', res)
        # 좋은 매칭점의 queryIdx로 원본 영상의 좌표 구하기 ---③
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
    # 좋은 매칭점의 trainIdx로 대상 영상의 좌표 구하기 ---④
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])

    return src_pts, dst_pts


def start(img1, img2):
    print("img1, img2", img1, ", ", img2)
    response1 = requests.get(img1)
    response2 = requests.get(img2)
    img1 = np.array(Image.open(io.BytesIO(response1.content)))
    img2 = np.array(Image.open(io.BytesIO(response2.content)))

    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    print("pre size : ", height1,",", width1,",", height2,",", width2)

    if(height1 < height2):
        height2 = height1
    else:
        height1 = height2
    if(width1 < width2):
        width2 = width1
    else:
        width1 = width2

    print("cur size : ", height1,",", width1,",", height2,",", width2)

    img1 = cv2.resize(img1, (int(width1), int(height1)), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (int(width2), int(height2)), interpolation=cv2.INTER_AREA)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, diff = compare_ssim(gray1, gray2, full=True)

    # full=True: 이미지 전체에 대해서 구조비교를 수행한다.
    diff = (diff * 255).astype('uint8')
    print(f'SSIM1 : {score:.6f}')


    src_pts, dst_pts = compare_sift_bfmatch(img1, img2, gray1, gray2)

    rect1 = cv2.boundingRect(src_pts)
    rect2 = cv2.boundingRect(dst_pts) # returns (x,y,w,h) of the rect
    img1 = img1[rect1[1]: rect1[1] + rect1[3], rect1[0]: rect1[0] + rect1[2]]
    img2 = img2[rect2[1]: rect2[1] + rect2[3], rect2[0]: rect2[0] + rect2[2]]


    img1 = cv2.resize(img1, (int(width1), int(height1)), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (int(width2), int(height2)), interpolation=cv2.INTER_AREA)
    #img2 = cv2.polylines(img2,[np.int32(dst_pts)],True,255,1, cv2.LINE_AA)
    #img2 = cv2.drawContours(img2, [np.int32(dst_pts)], 0, (0, 255, 0), 3)
    
    grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, diff = compare_ssim(grayA, grayB, full=True)


    res = cv2.drawMatches(img1, [], img2, [], [], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)



    # full=True: 이미지 전체에 대해서 구조비교를 수행한다.
    diff = (diff * 255).astype('uint8')
    print(f'SSIM3 : {score:.6f}')

    cv2.imshow('SSIM + resize', res)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    pass