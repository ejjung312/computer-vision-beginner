import os
import argparse

import cv2

# DNN 모델
# CNN 기반 SSD(Single Shot MultiBox Detector) 아키텍처
model_path = "opencv_face_detector_uint8.pb" 
config_path = "opencv_face_detector.pbtxt"

# DNN 로드
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

# 동영상 읽기
cap = cv2.VideoCapture("video3.mp4")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 프레임 크기 가져오기
    h, w = frame.shape[:2]
    
    # 프레임을 blob 형식으로 변환
    blob = cv2.dnn.blobFromImage(frame, 
                                scalefactor=1.0, # 픽셀 값 조정 (1.0으로 원본 유지)
                                size=(300, 300), # DNN 모델 입력 크기. 300,300 고정 크기 사용
                                mean=(104, 117, 123), # 평균값을 빼서 데이터 정규화
                                swapRB=False, # RGB <-> BGR 변환 여부
                                crop=False) # 입력 이미지 자르기 여부
    
    # DNN 네크워크에 입력
    net.setInput(blob)
    
    detections = net.forward()
    
    for i in range(detections.shape[2]): # 검출된 객체(얼굴)의 수
        confidence = detections[0, 0, i, 2] # 신뢰도 추출
        if confidence > 0.5: # 임계값
            box = detections[0, 0, i, 3:7] * [w, h, w, h] # 좌표는 비율로 제공되므로 w, h를 곱하여 실제 픽셀 좌표로 변환
            (x1, y1, x2, y2) = box.astype("int")
            
            # 얼굴 영역에 사각형 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = f"{confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # 결과 화면 표시
    cv2.imshow("Face Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cap.destroyAllWindows()