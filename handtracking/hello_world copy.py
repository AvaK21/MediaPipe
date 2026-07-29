import os
import cv2
import time
from collections import namedtuple
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#Look up python naming conventions next and then finish the video

# Based on the Hand Landmarker example from the MediaPipe documentation: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python
# and previous work I have done with the MediaPipe Gesture Recognizer

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH,"..","hand_landmarker.task")


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResults = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# h, w are normalized to [0,1] of frame size so  lm.x * w and lm.y * h gives the pixel coordinates of the landmark. 
# cv2.line and cv2.circle expect int coordinates

FingerGroup = namedtuple('FingerGroup', ['name','connections', 'color', 'highlight_color'])

FINGER_GROUPS = [
    FingerGroup(
        name='palm',
        connections=[(0,1),(2,5),(0,17),(5,9),(9,13),(13,17)],
        color=(180, 180, 180), 
        highlight_color=(0, 0, 220)
        ),
    FingerGroup(
        name='thumb',
        connections=[(1,2),(2,3),(3,4)],          
        color=(120, 190, 210), 
        highlight_color=(120, 190, 210)
        ),
    FingerGroup(
        name='index',
        connections=[(5,6),(6,7),(7,8)],          
        color=(180,  60, 150), 
        highlight_color=(180,  60, 150)
        ),
    FingerGroup(
        name='middle',
        connections=[(9,10),(10,11),(11,12)],     
        color=(0,210, 220), 
        highlight_color=(0,   210, 220)
        ),
    FingerGroup(
        name='ring',
        connections=[(13,14),(14,15),(15,16)],    
        color=(0,   200,  80), 
        highlight_color=(0,   200,  80)),
    FingerGroup(
        name='pinky',
        connections=[(17,18),(18,19),(19,20)],    
        color=(220, 100,   0), 
        highlight_color=(220, 100,   0)),
]

def draw_landmarks(frame, hand_landmarks):
    h, w = frame.shape[:2]

    #List comprehension - in this case result list of tuples of (x,y) coordinates of the landmarks in pixel coordinates.
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for group in FINGER_GROUPS:
        for a, b in group.connections:
            cv2.line(frame, pts[a], pts[b], (100,100,100), 2) # Colors in BGR
            cv2.circle(frame, pts[a], 6, (200,0,200),       -1) # Thickness -1 means filled circle
            cv2.circle(frame, pts[a], 6, (255, 255, 255),  1)
            cv2.circle(frame, pts[b], 6, (200,0,200),       -1)
            cv2.circle(frame, pts[b], 6, (255, 255, 255),  1)


def main():
    latest_result = [None] #mutable list - needs a value on first call

    #Function that is called when the hand landmarker has a result to return. This function is called in a separate thread, so it is asynchronous. It is called with the result, the output image, and the timestamp in milliseconds.
    def on_detection_result(result: HandLandmarkerResults, output_image: mp.Image, timestamp_ms: int):
        #print('hand landmarker result: {}'.format(result))
        latest_result[0] = result #store the latest result in the mutable list

    #Create a hand lamdmaker blueprint (instantiate) with the live stream mode:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH), # Where the model is located
        running_mode=VisionRunningMode.LIVE_STREAM, # Mode
        result_callback=on_detection_result, #Function to call when a result is available
        num_hands=2, #max number of hands to detect, default is 1
        min_hand_detection_confidence=0.5, #minimum confidence for hand detection, default is 0.5
        min_hand_presence_confidence=0.5, #minimum confidence for hand presence, default is 0.5
        min_tracking_confidence=0.5, #minimum confidence for hand tracking, default is 0.5
    )


    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Press q on the cv2 window to exit")
    last_ts = 0

    #Instantate the hand landmarker with loading the .task model in memory, and create the object - landmarker
    with HandLandmarker.create_from_options(options) as landmarker:


        
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)




            #cv2 is in BGR, mediapipe needs RGB, CONVERT
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


            # Version that MediaPipe Hand Tracker uses, which is a mediapipe Image object. This is the format that the Hand Landmarker expects.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)


            # If the timestamp is not greater than the previous timestamp, increment it by 1 to ensure that the timestamps are monotonically increasing.
            # Otherwise the program will almost always crash immediately:
            ts = int(time.time() * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts

            landmarker.detect_async(mp_image, timestamp_ms=ts)

            if latest_result[0] is not None and latest_result[0].hand_landmarks:
                for hand_landmarks in latest_result[0].hand_landmarks:
                    draw_landmarks(frame, hand_landmarks)


            cv2.imshow("Hand Tracking", frame)

            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()