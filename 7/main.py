from ultralytics import YOLO
import cv2

# yolov8 로드
model = YOLO('yolov8n.pt')

video_path = './test.mp4'
cap = cv2.VideoCapture(video_path)

ret = True
while ret:
    ret, frame = cap.read()
    
    if ret:
        results = model.track(frame, persist=True)
        
        # 프레임에 결과 플롯
        frame_ = results[0].plot()
        
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break