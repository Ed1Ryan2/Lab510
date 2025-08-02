import cv2
 
cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

if not cap.isOpened():
    print(99999)
    exit()
    
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("无法获取图像")
        break
    
    cv2.imshow("Camera Feed", frame)
    
    if cv2.waitKey(10)&0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()