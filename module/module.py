"""
    About see how to tell if a finger is extended or not 

    Ideas : if 8 is above 6, but only would work for pure vertical hand,
    maybe a dot product between vectors 0 and 5? and nuckle and tip of finger to see if in same direction and then 
    check if tip is more in than direction than the nuckle, if so then extended, if not then not extended.
    also for thumb if extended tip os past point or 2 ponts  opposite direction of the hand ( need to check/ account for if left or right hand, hand if flipped, hand is upside down or sideways , 90 off

    maybe disable the count if hand is perperdicular to camera or close ( y position of wrist and middle finger tip are close together, or if the hand is upside down ( y position of wrist is above y position of middle finger tip)
    (0,0)      (max,0)
    
    (0,max)   (max,max)
"""
# https://www.youtube.com/watch?v=p5Z_GGRCI5s

# index to pinky
# DPT angle is above 150, if vector D to T is in same direction  (roughly (will need to test) as wrist to 9) through dot product, then extended, if not then not extended.

# thumb is more complicated,
# first see if hand is left or right, and if flipped, then check if tip is past 1 point below or 2 (need to test)

# Later is designated program or file, calculate all the angles once and then let function all use same data so only measured once, and so there isn't desicremence across data

#TODO: As source flip the handded because the camera is flipped. SO it is correct for later functions
import os
import cv2
import time
import numpy as np
import mediapipe as mp
import math
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark, HandLandmarksConnections

#LLook up the gesture model math if that is available. 

# Based on the Hand Landmarker example from the MediaPipe documentation: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python
# and legacy solutions tutorial, and AI
# and previous work I have done with the MediaPipe Gesture Recognizer



JOINT_LIST = [ 
    [HandLandmark.THUMB_MCP,HandLandmark.THUMB_IP,HandLandmark.THUMB_TIP],
    [HandLandmark.INDEX_FINGER_PIP,HandLandmark.INDEX_FINGER_DIP,HandLandmark.INDEX_FINGER_TIP], 
    [HandLandmark.MIDDLE_FINGER_PIP,HandLandmark.MIDDLE_FINGER_DIP,HandLandmark.MIDDLE_FINGER_TIP],
    [HandLandmark.RING_FINGER_PIP,HandLandmark.RING_FINGER_DIP,HandLandmark.RING_FINGER_TIP], 
    [HandLandmark.PINKY_PIP,HandLandmark.PINKY_DIP,HandLandmark.PINKY_TIP], 
    ]
EXTENED_FINGER_THRESHOLD = 160  # Angle threshold to consider a finger as extended
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResults = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# h, w are normalized to [0,1] of frame size so  lm.x * w and lm.y * h gives the pixel coordinates of the landmark. 
# cv2.line and cv2.circle expect int coordinates

class HandAnalyzer:
    """
    Full hand analysis pipeline,
    Owns the MediaPipe HandLandmarker model and exposes methods
    """

    EXTENDED_THRESHOLD = 160

    def __init__(self, model_path: str, num_hands : int = 2):
        self.model_path = model_path
        self.num_hands = num_hands
        self._latest_result = None
        self._mp_image = None  # Store the latest MediaPipe image for drawing
  

        # Create a hand landmarker with the specified options
        self.options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._on_result,
            num_hands=self.num_hands,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = HandLandmarker.create_from_options(self.options)

    def process_frame(self, frame: np.ndarray, timestamp_ms)-> None:
        """
        Process a single frame 
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._landmarker.detect_async(self._mp_image, timestamp_ms=timestamp_ms)

    def analyze_results(self,frame) -> tuple[list[list[bool]], int] | None:
        """
        Analyze the latest results and draw on the frame
        """
        if self._latest_result is not None and self._latest_result.hand_landmarks:
            zipped = zip(
                self._latest_result.hand_landmarks, 
                self._latest_result.handedness, 
            )
            extended_fingers = []
            for hand_landmark, handed in zipped:
                handed = handed[0]  # Get the handedness for the current hand (out of the category list)
                hand_label = self.flip_hand_name(handed.display_name)  # Flip the handedness name if needed
                hand_score = handed.score
                self.draw_landmarks(frame, hand_landmark, hand_label, hand_score)
                self.draw_angles(frame, hand_landmark, JOINT_LIST)
                extended_fingers.append(self.is_extended_fingers(hand_landmark, hand_label))
            num_extended = self.count_extended_fingers(extended_fingers)
            self.draw_num_extended_fingers(frame, num_extended)
            return extended_fingers, num_extended
        else: 
            return None
    

    def _on_result(self, result, output_image: mp.Image, timestamp_ms: int):
        #print('hand landmarker result: {}'.format(result))
         #use the mutable list to store the latest result
        self._latest_result = result #store the latest result in the mutable list

    def close(self):
        self._landmarker.close()

    #assume is 1 hand
    def draw_landmarks(self, frame, hand_landmark, hand_label, hand_score):
        h, w = frame.shape[:2]
        #List comprehension - in this case result list of tuples of (x,y) coordinates of the landmarks in pixel coordinates.
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmark]
        for connection in HandLandmarksConnections.HAND_CONNECTIONS:
            a, b = connection.start, connection.end
            cv2.line(frame, pts[a], pts[b], (100,100,100), 2) # Colors in BGR

        for pt in pts:
            cv2.circle(frame, pt, 6, (200,0,200), cv2.FILLED)
            cv2.circle(frame, pt, 6, (255, 255, 255), 1)


    
        text = f"{hand_label} ({hand_score:.2f})"

        coord = (pts[HandLandmark.WRIST][0], pts[HandLandmark.WRIST][1] + 20)  # Position the text below the wrist landmark
        cv2.putText(frame, text,coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # cos θ = (a·b) / (|a| |b|) ---this case--> θ = arccos((ba @ bc) / (||ba|| * ||bc||)) 




    def draw_angles(self, frame, hand_landmark, joint_list):
        """
        Draws the angle value at the middle joint of each triplet.
        Uses image coords (2D) for the math and draw position.
        hand_world_landmark and (3D) tested and dropped -- confirmed unreliaable in bug #5571
        """
        h, w = frame.shape[:2]
        for joint in joint_list:
            a, b, c = joint
            angle = self.angle_between(
                hand_landmark[a], 
                hand_landmark[b], 
                hand_landmark[c]
            )

            # Draw position of the middle joint in image coordinates
            pt = (int(hand_landmark[b].x * w), int(hand_landmark[b].y * h))
            cv2.putText(frame, f"{angle:.0f}", pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 2, cv2.LINE_AA)

    def is_extended_fingers(self, hand_landmark, hand_label):
        extended_fingers = []

        # Check each finger (index, middle, ring, pinky)
        for finger in [HandLandmark.INDEX_FINGER_TIP, HandLandmark.MIDDLE_FINGER_TIP, HandLandmark.RING_FINGER_TIP, HandLandmark.PINKY_TIP]:
            angle = self.angle_between(
                hand_landmark[finger - 2],  # PIP joint
                hand_landmark[finger - 1],  # DIP joint
                hand_landmark[finger]       # Tip
            )
            dot = self.get_dot_product(hand_landmark[HandLandmark.WRIST], hand_landmark[finger - 2], hand_landmark[finger])
            print(f"Finger {finger}: Angle = {angle:.2f}, Dot Product = {dot:.2f}")
            if dot < 0 and angle > EXTENED_FINGER_THRESHOLD:  # Threshold for extended finger
                extended_fingers.append(True)
            else:
                extended_fingers.append(False)
        thumb =self.thumb_extended_check(hand_landmark, hand_label)
        extended_fingers.append(thumb)

        print(f"Extended fingers for {hand_label}: {extended_fingers}")
        return extended_fingers

    def count_extended_fingers(self, extended_fingers: list) -> int:
        if extended_fingers is None:
            return 0
        #Normailze to list of lists
        if not isinstance(extended_fingers[0], list):
            extended_fingers = [extended_fingers]
        num_extended = sum(1 for hand in extended_fingers for extended in hand if extended)
        return num_extended


    def draw_num_extended_fingers(self,frame, num_extended) -> None:


        text = f"Count: {num_extended}"
        print(num_extended)

        cv2.putText(frame, text, (250,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,0,200), cv2.LINE_AA)



    def flip_hand_name(self, name:str) -> str:
        return "Right" if name == "Left" else "Left"

    def get_dot_product(self, a,b,c):
        """
        a: handlamrtk. enum
        b: handlamrtk. enum
        c: handlamrtk. enum
        returns the dot product of the two vectors, which is a measure of how much they point in the same direction.

        Ex: Wrist to DIP of a finger, and DIP to TIP of the same finger. If the dot product is positive...
        """
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        # make b the origin of the vectors by subtracting b from a and c
        ba = a - b
        bc = c -b

        return np.dot(ba, bc) 


    def is_palm_facing_camera(self, hand_landmark, hand_label):
        """
        Infer palm direction from landmark winding order.
        Returns True if palm is facing camera.
        """
        mid    = np.array([hand_landmark[HandLandmark.MIDDLE_FINGER_MCP].x, hand_landmark[HandLandmark.MIDDLE_FINGER_MCP].y]) 
        wrist  = np.array([hand_landmark[HandLandmark.WRIST].x, hand_landmark[HandLandmark.WRIST].y]) 
        pinky  = np.array([hand_landmark[HandLandmark.PINKY_MCP].x, hand_landmark[HandLandmark.PINKY_MCP].y])

        cross = np.cross(mid - wrist, pinky - wrist)

        if cross == 0:
            return False  # Palm is not facing camera if the cross product is zero

        if hand_label == "Right":
            return cross > 0
        else:
            return cross < 0
        
    def thumb_extended_check(self,hand_landmark, hand_label):
        index_mcp = np.array([hand_landmark[HandLandmark.INDEX_FINGER_MCP].x, hand_landmark[HandLandmark.INDEX_FINGER_MCP].y])
        thumb_mcp = np.array([hand_landmark[HandLandmark.THUMB_MCP].x, hand_landmark[HandLandmark.THUMB_MCP].y])
        thumb_tip = np.array([hand_landmark[HandLandmark.THUMB_TIP].x, hand_landmark[HandLandmark.THUMB_TIP].y])

        cross = np.cross( index_mcp - thumb_mcp, thumb_tip - thumb_mcp)


        palm_facing = self.is_palm_facing_camera(hand_landmark, hand_label)

        if cross == 0 or palm_facing is None:
            return False  # Thumb is not extended if the cross product is zero or palm is not facing camera

        if hand_label == "Right":
            expected_sign = cross < 0
        else:
            expected_sign = cross > 0

        # Flip logic if palm is facing away
        if not palm_facing:
            expected_sign = not expected_sign
        result = bool(expected_sign)
        return result




    def distance(self,a,b):
        """
        a: handlamrtk. enum
        b: handlamrtk. enum

        returns: the distance between two points in 2D space. Euclidean distance is calculated using numpy's linear algebra norm function.
        Is the same as math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) but more efficient and concise.
        """
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        return np.linalg.norm(a - b)


    def angle_between(self, a, b, c):
        """
        Calculate the angle between three points a, b, and c in 2D space.
        """
        # Convert the points to numpy arrays (vectors)
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        # Remove the effect of the middle point b by translating the points so that b is at the origin
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        # Radians, clips because arcos expects input in [-1, 1] (because cos only outputs [-1,1])
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)