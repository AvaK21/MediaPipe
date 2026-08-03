import os
import cv2
import time
from module import HandAnalyzer

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH,"..","Models", "hand_landmarker.task")

def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Press q on the cv2 window to exit")
    last_ts = 0
    p_time = time.time()

    analyzer = HandAnalyzer(MODEL_PATH)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        # If the timestamp is not greater than the previous timestamp, increment it by 1 to ensure that the timestamps are monotonically increasing.
        # Otherwise the program will almost always crash immediately:
        ts = int(time.time() * 1000)
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts

        #landmarker.detect_async(mp_image, timestamp_ms=ts)
        analyzer.process_frame(frame, ts)

        results  = analyzer.analyze_results(frame) if analyzer.analyze_results(frame) else (None, None)
        num_extended = num_extended if results is not None and  results[1] is not None else 0
        print(f"Extended: {num_extended} fingers detected.")
        


        

        c_time = time.time()
        fps = 1 / (c_time - p_time ) if p_time != 0 else 0
        p_time = c_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Hand Tracking", frame)

        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    analyzer.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()