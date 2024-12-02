import numpy as np
import cv2


def get_limits(color):
    c = np.uint8([[color]])  # BGR 값을 numpy 배열로 만듬
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV) # BGR을 HSV로 변환

    hue = hsvC[0][0][0]  # hue 값 추출

    if hue >= 165:  # 빨간색 상한에 가까운 경우
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:  # 빨간색 하한에 가까운 경우
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else: # 일반적인 경우
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit