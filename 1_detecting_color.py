import cv2
from PIL import Image

from util import get_limits

# 동영상 파일 경로 설정
video_path = "video2.mp4"

# 동영상 파일 열기
cap = cv2.VideoCapture(video_path)

yellow = [0, 255, 255]  # yellow in BGR colorspace
if not cap.isOpened():
    print("동영상을 열 수 없습니다.")
else:
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        
        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lowerLimit, upperLimit = get_limits(color=yellow)

        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

        mask_ = Image.fromarray(mask)

        bbox = mask_.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        
        # 프레임을 화면에 표시
        cv2.imshow("Frame", frame)
        
        # 키보드 입력 대기 (30ms)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

# 자원 해제
cap.release()
cv2.destroyAllWindows()