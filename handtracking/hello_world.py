import os
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Based on the Hand Landmarker example from the MediaPipe documentation: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python
# and previous work I have done with the MediaPipe Gesture Recognizer

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH,"..","hand_landmarker.task")


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResults = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def main():
    #Function that is called when the hand landmarker has a result to return. This function is called in a separate thread, so it is asynchronous. It is called with the result, the output image, and the timestamp in milliseconds.
    def print_result(result: HandLandmarkerResults, output_image: mp.Image, timestamp_ms: int):
        print('hand landmarker result: {}'.format(result))

    #Create a hand lamdmaker blueprint (instantiate) with the live stream mode:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH), # Where the model is located
        running_mode=VisionRunningMode.LIVE_STREAM, # Mode
        result_callback=print_result, #Function to call when a result is available
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
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # If the timestamp is not greater than the previous timestamp, increment it by 1 to ensure that the timestamps are monotonically increasing.
            # Otherwise the program will almost always crash immediately:
            ts = int(time.time() * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts

            landmarker.detect_async(mp_image, timestamp_ms=ts)
            cv2.imshow("Hand Tracking", frame)


            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()