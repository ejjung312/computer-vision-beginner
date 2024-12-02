import os
import argparse

import cv2

# Haar Cascade 모델 로드
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# 동영상 읽기
cap = cv2.VideoCapture("video3.mp4")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 검출
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    
    # 검출된 얼굴에 블러처리
    for (x, y, w, h) in faces:
        # 얼굴 ROI 추출
        face_roi = frame[y:y+h, x:x+w]
        
        # 블러 처리
        face_roi_blurred = cv2.GaussianBlur(face_roi, (25, 25), 30)
        
        # 블러 처리한 영역을 원본 프레임에 반영
        frame[y:y+h, x:x+w] = face_roi_blurred
    
    # 결과 화면 표시
    cv2.imshow("Face Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cap.destroyAllWindows()