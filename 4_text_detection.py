import cv2
import numpy as np

# EAST 모델 파일 경로 설정
model_path = "./data/4/frozen_east_text_detection.pb"

# 이미지 읽기
image = cv2.imread("./data/4/test3.png")
orig = image.copy()
(H, W) = image.shape[:2]

# 네트워크 입력 크기 설정
(newW, newH) = (320, 320)
rW = W / float(newW)
rH = H / float(newH)

# 이미지를 네트워크 입력 크기로 변환
blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)

# EAST 모델 로드
net = cv2.dnn.readNet(model_path)

# 네트워크에 입력 설정
net.setInput(blob)

# 관심 레이어 설정
layerNames = [
    "feature_fusion/Conv_7/Sigmoid", # 신뢰도
    "feature_fusion/concat_3" # 기하학적 데이터
]

# 순방향 패스 실행
(scores, geometry) = net.forward(layerNames)

# 결과 처리 함수
def decode_predictions(scores, geometry, confThreshold=0.5):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    
    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        
        for x in range(numCols):
            if scoresData[x] < confThreshold:
                continue
                
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)

# 텍스트 영역 검출
(rects, confidences) = decode_predictions(scores, geometry)

# 중복 경계 상자 제거
boxes = cv2.dnn.NMSBoxes(rects, confidences, 0.5, 0.4)

if len(boxes) > 0:
    for i in boxes.flatten():
        (startX, startY, endX, endY) = rects[i]

        # 원래 크기에 맞게 조정
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # 경계 상자 그리기
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

# 결과 출력
cv2.imshow("Text Detection", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()