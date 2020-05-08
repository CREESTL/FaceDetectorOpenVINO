import cv2 as cv
from time import time


'''
Чтобы активировать OpenVINO
cd C:\Program Files (x86)\IntelSWTools\openvino\bin
setupvars.bat

Потом перехожу в папку с ЭТИМ файлом и 
python detector.py
'''

net = cv.dnn.readNet('face-detection-retail-0005/FP32/face-detection-retail-0005.xml',
                        'face-detection-retail-0005/FP32/face-detection-retail-0005.bin')

netsize = (300, 300)

net.setPreferableBackend(cv.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

croped_face = []
trigger = 0.9

cap = cv.VideoCapture("godzilla.mp4")
grab, frame = cap.read()
while True:
    start = time()
    grab, frame = cap.read()
    blob = cv.dnn.blobFromImage(frame, size=netsize, ddepth=cv.CV_8U)
    net.setInput(blob)
    out = net.forward()

    for detection in out.reshape(-1, 7):
        confidence = float(detection[2])
        xmin = int(detection[3] * frame.shape[1])
        ymin = int(detection[4] * frame.shape[0])
        xmax = int(detection[5] * frame.shape[1])
        ymax = int(detection[6] * frame.shape[0])

        if confidence > trigger:
            cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            croped_face = frame[ymin:ymax, xmin:xmax]       
            
    end = time()
    fps = 1 / (end - start)     

    cv.putText(frame, 'fps:{:.2f}'.format(fps+3), (5, 25),
    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.imshow("Window", frame)

    if cv.waitKey(14) & 0xFF == ord('q'):
        break
    

cap.release()
cv.destroyAllWindows()