import ctypes
import mediapipe as mp
import cv2
import time
import numpy as np
import math
from ctypes import cast, POINTER, pointer
from comtypes import CLSCTX_ALL
from pycaw.pycaw import *

wCam, hCam = 900, 650 # Dimensions of camera window.
ptime = 0
cap = cv2.VideoCapture(0)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
vc = cast(interface, POINTER(IAudioEndpointVolume))
Range = vc.GetVolumeRange()
minR = Range[0]
maxR = Range[1]
volBar = 400
volPer = 0
mpHands = mp.solutions.hands
hands = mpHands.Hands() 
mpDraw = mp.solutions.drawing_utils
cap.set(3, wCam)
cap.set(4, hCam)

while True:
    ret, frame = cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB) #Process method will process resullt and give output.
    lmlist = []
    if results.multi_hand_landmarks:
        for hand_in_frame in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_in_frame, mpHands.HAND_CONNECTIONS)
        
        for id, lm in enumerate(results.multi_hand_landmarks[0].landmark):
            h, w, c = frame.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            lmlist.append([cx, cy])

        if len(lmlist) != 0:
            x1, y1 = lmlist[4][0], lmlist[4][1]
            x2, y2 = lmlist[8][0], lmlist[8][1] 

            #Above 2 lines find out the co-ordinates of the thumb and index finger 

            cv2.circle(frame, (x1, y1), 14, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 14, (255, 0, 255), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 3, cv2.FILLED)
            length = math.hypot(x2-x1, y2-y1) #Length of the line drawn from thumb to index finger.

            vol = np.interp(length, [50, 300], [minR, maxR])
            print(vol)
            vc.SetMasterVolumeLevel(vol, None)
            volBar = np.interp(length, [50, 300], [400, 150])
            volPer = np.interp(length, [50, 300], [0, 100])
            

            cv2.rectangle(frame, (50, 150), (85, 400), (255, 0, 255))
            cv2.rectangle(frame, (50, int(volBar)), (85, 400), (255, 0, 255), cv2.FILLED)     

            cv2.putText(frame, f'{int(volPer)} %', (85, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

    
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(frame, str(int(fps)),(10,70), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 255), 3)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord("x"):
        break
cap.release()
cv2.destroyAllWindows()
