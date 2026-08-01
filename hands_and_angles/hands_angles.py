import os
import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark, HandLandmarksConnections

#LLook up the gesture model math if that is available. 

# Based on the Hand Landmarker example from the MediaPipe documentation: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python
# and legacy solutions tutorial, and AI
# and previous work I have done with the MediaPipe Gesture Recognizer

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH,"..","Models", "hand_landmarker.task")
JOINT_LIST = [ [6,7,8], [10,11,12],[14,15,16], [18,19,20], [2,3,4]]

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResults = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# h, w are normalized to [0,1] of frame size so  lm.x * w and lm.y * h gives the pixel coordinates of the landmark. 
# cv2.line and cv2.circle expect int coordinates

#assume is 1 hand
def draw_landmarks(frame, hand_landmark, handed):
    h, w = frame.shape[:2]
    #List comprehension - in this case result list of tuples of (x,y) coordinates of the landmarks in pixel coordinates.
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmark]
    for connection in HandLandmarksConnections.HAND_CONNECTIONS:
        a, b = connection.start, connection.end
        cv2.line(frame, pts[a], pts[b], (100,100,100), 2) # Colors in BGR

    for pt in pts:
        cv2.circle(frame, pt, 6, (200,0,200), -1)
        cv2.circle(frame, pt, 6, (255, 255, 255), 1)

    name = handed.display_name
    score = handed.score
    name = "Right" if name == "Left" else "Left"
    text = f"{name} ({score:.2f})"

    coord = (pts[HandLandmark.WRIST][0], pts[HandLandmark.WRIST][1] + 20)  # Position the text below the wrist landmark
    cv2.putText(frame, text,coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

# Learn when to do python math or with np 
# def draw_angle(frame, hand_world_landmark, JOINT_LIST):
#     print("hand_world_landmark", hand_world_landmark)
    # h, w = frame.shape[:2]
    # for joint in JOINT_LIST:
    #    a, b, c = joint
    #    pt_a = np.array([hand_world_landmark[a].x * w, hand_world_landmark[a].y * h])
    #    pt_b = np.array([hand_world_landmark[b].x * w, hand_world_landmark[b].y * h])
    #    pt_c = np.array([hand_world_landmark[c].x * w, hand_world_landmark[c].y * h])



def main():
    latest_result = None #mutable list - needs a value on first call

    #Function that is called when the hand landmarker has a result to return. This function is called in a separate thread, so it is asynchronous. It is called with the result, the output image, and the timestamp in milliseconds.
    def on_detection_result(result, output_image: mp.Image, timestamp_ms: int):
        #print('hand landmarker result: {}'.format(result))
        nonlocal latest_result #use the mutable list to store the latest result
        latest_result = result #store the latest result in the mutable list

    #Create a hand lamdmaker blueprint (instantiate) with the live stream mode:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH), # Where the model is located
        running_mode=VisionRunningMode.LIVE_STREAM, # Mode
        result_callback=on_detection_result, #Function to call when a result is available
        num_hands=2, #max number of hands to detect, default is 1
        min_hand_detection_confidence=0.5, #minimum confidence for hand detection, default is 0.5
        min_hand_presence_confidence=0.5, #minimum confidence for hand presence, default is 0.5
        min_tracking_confidence=0.7, #minimum confidence for hand tracking, default is 0.5
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

            # Draw landmarks for each hand
            if latest_result is not None and latest_result.hand_landmarks:
                zipped = zip(
                    latest_result.hand_landmarks, 
                    latest_result.handedness, 
                    latest_result.hand_world_landmarks
                    )
                for hand_landmark,handed, hand_world_landmark in zipped:
                    # handed = latest_result[0].handedness[i][0] # Get the handedness for the current hand
                    handed = handed[0]  # Get the handedness for the current hand (out of the category list)

                    draw_landmarks(frame, hand_landmark, handed)
                    #TODO draw_angle(frame, hand_world_landmark, JOINT_LIST)


            cv2.imshow("Hand Tracking", frame)

            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()