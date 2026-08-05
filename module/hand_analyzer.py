"""
Hand Analyzer Module
=========================
By: Ava K. < 

Detects hand landmarks via MediaPipe's Hand Landmarker and determines which fingers are extended.

Extension logic:
    - Index, middle, ring, pinky: PIP-DIP-TIP angle > threshold AND
      dot product of (wrist→PIP) · (PIP→TIP) < 0 (finger pointing away from wrist).
    - Thumb: cross product of (thumb_mcp→index_mcp) × (thumb_mcp→tip) determines
      which side of the knuckle line the tip falls on, corrected for handedness and
      palm orientation.
 
Coordinate system (MediaPipe normalized):
    (0,0) = top-left      (1,0) = top-right
    (0,1) = bottom-left   (1,1) = bottom-right

Comments:
    -  h, w are normalized to [0,1] of frame size so  lm.x * w and lm.y * h gives the pixel coordinates of a landmark. 
    -  cv2.line and cv2.circle expect int coordinates
 
References:
    - MediaPipe Hand Landmarker: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python
    - Bug #5571: hand_world_landmark (3D) confirmed unreliable; 2D image coords used throughout.
 
"""
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark, HandLandmarksConnections

# --- MediaPipe aliases ---

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResults = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# --- Constants ---

# Triplets (a,b,c), b is vertex, used to calculate angles used to determine if fingers are extended
JOINT_LIST = [ 
    [HandLandmark.THUMB_MCP,            HandLandmark.THUMB_IP,          HandLandmark.THUMB_TIP],
    [HandLandmark.INDEX_FINGER_PIP,     HandLandmark.INDEX_FINGER_DIP,  HandLandmark.INDEX_FINGER_TIP], 
    [HandLandmark.MIDDLE_FINGER_PIP,    HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.MIDDLE_FINGER_TIP],
    [HandLandmark.RING_FINGER_PIP,      HandLandmark.RING_FINGER_DIP,   HandLandmark.RING_FINGER_TIP], 
    [HandLandmark.PINKY_PIP,            HandLandmark.PINKY_DIP,         HandLandmark.PINKY_TIP], 
    ]

# Minimum PIP-DIP-TIP angle (degrees) for the possibility that a finger is extended.
EXTENED_FINGER_THRESHOLD = 170 

# --- Pure math utilities ---

def lm_to_vec(lm) -> np.ndarray:
    """Convert a MediaPipe landmark to a 2D numpy vector."""
    return np.array([lm.x, lm.y])



def flip_hand_name(name:str) -> str:
    """ Flip Left <-> Right to comidate for mirrored camera image. """
    return "Right" if name == "Left" else "Left"

    


def angle_between(a, b, c) -> float:
    """
    Angle at vertex b formed by points a-b-c (degrees).
 
    Uses: θ = arccos((ba · bc) / (|ba| |bc|))
    Clips cosine to [-1, 1] before arccos to guard against floating-point drift.
    """
    # Convert the points to numpy arrays (vectors)
    a = lm_to_vec(a)
    b = lm_to_vec(b)
    c = lm_to_vec(c)

    # Remove the effect of the middle point b by translating the points so that b is at the origin
    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Radians, clips because arcos expects input in [-1, 1] (because cos only outputs [-1,1])
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)


def dot_product(a,b,c) -> float:
        """
        a: handlamrtk. enum
        b: handlamrtk. enum
        c: handlamrtk. enum

        Returns: Dot product of vector b->a and vector b->c. 
        
        Positive dot product: vectors point in the same general direction (angle < 90 degrees).
        Negative dot product: vectors point in opposite directions (angle > 90 degrees).


        Ex: Wrist to DIP of a finger, and DIP to TIP of the same finger. 
        If the dot product is negative, it indicate the finger is extended away from the wrist.
        """
        a = lm_to_vec(a)
        b = lm_to_vec(b)
        c = lm_to_vec(c)

        # make b the origin of the vectors by subtracting b from a and c
        ba = a - b
        bc = c -b

        return np.dot(ba, bc) 

# --- Main HandAnalyzer class ---

class HandAnalyzer:
    """
    Full hand analysis pipeline,
    Owns the MediaPipe HandLandmarker model and exposes methods

    Call procress_frame() each loop iteration, then analyze_results() to get the extended finger count and draw on the frame.
    
    Args:
        model_path: Path to the hand_landmarker.task model file.
        num_hands: Maximum number of hands to detect (default 2).
    
    """

    EXTENDED_THRESHOLD = 160

    def __init__(self, model_path: str, num_hands : int = 2):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path
        self.num_hands = num_hands
        self._latest_result = None
  

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
        Process a single frame: BGR cv2 image, convert to RGB, and send to the hand landmarker for asynchronous detection.

        Args:
            frame: BGR image from cv2.VideoCapture.
            timestamp_ms: Timestamp in milliseconds for the frame (must be monotonically increasing).
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._landmarker.detect_async(mp_image, timestamp_ms=timestamp_ms)

    def analyze_results(self,frame) -> tuple[list[list[bool]], int] | None:
        """
        Analyze the latest results and draw overlays on the frame.

        Args:
            frame: BGR image from cv2.VideoCapture to draw landmarks, angles, and extended finger count.
        
        Returns:
            (extended_fingers, num_extended) where extended_fingers is a list of
            per-hand bool lists [index, middle, ring, pinky, thumb], or None if
            no hands are detected.
        """
        if self._latest_result is not None and self._latest_result.hand_landmarks:
            zipped = zip(
                self._latest_result.hand_landmarks, 
                self._latest_result.handedness, 
            )
            extended_fingers = []
            for hand_landmark, handed in zipped:
                handed = handed[0]  # Get the handedness for the current hand (out of the category list)
                hand_label = flip_hand_name(handed.display_name)  # Flip the handedness name if needed
                hand_score = handed.score
                self._draw_landmarks(frame, hand_landmark, hand_label, hand_score)
                self._draw_angles(frame, hand_landmark, JOINT_LIST)
                extended_fingers.append(self._is_extended(hand_landmark, hand_label))
            num_extended = self._count_extended_fingers(extended_fingers)
            self._draw_extended_count(frame, num_extended)
            return extended_fingers, num_extended
        else: 
            return None
    

    def close(self) -> None:
        """Release resources associated with the hand landmarker. Call on exit."""
        self._landmarker.close()

    # --- Callback Methods ---

    def _on_result(self, result, output_image: mp.Image, timestamp_ms: int) -> None:
        """Runs on a seperate thread when the hand landmarker has a result. Stores the latest result in a mutable list."""
        #print('hand landmarker result: {}'.format(result))
        self._latest_result = result #store the latest result in the mutable list

    # --- Extended Logic ---

    
    def _is_extended(self, hand_landmark, hand_label) -> list[bool]:
        """
        Determine which fingers are extended for a single hand. 
        
        Args:
            hand_landmark: List of MediaPipe hand landmarks for a single hand.
            hand_label: String label for the hand ("Left" or "Right").

        Returns:
            list of bools [thumb, index, middle, ring, pinky].
        
        """
        extended_fingers = []

        #Check thumb separately because it has a different extension logic
        thumb =self._is_thumb_extended(hand_landmark, hand_label)
        extended_fingers.append(thumb)

        # Check fingers if extended (index, middle, ring, pinky)
        for finger in [HandLandmark.INDEX_FINGER_TIP, HandLandmark.MIDDLE_FINGER_TIP, HandLandmark.RING_FINGER_TIP, HandLandmark.PINKY_TIP]:
            angle = angle_between(
                hand_landmark[finger - 2],  # PIP joint
                hand_landmark[finger - 1],  # DIP joint
                hand_landmark[finger]       # Tip
            )
            dot = dot_product(hand_landmark[HandLandmark.WRIST], hand_landmark[finger - 2], hand_landmark[finger])
            #print(f"Finger {finger}: Angle = {angle:.2f}, Dot Product = {dot:.2f}")
            if dot < 0 and angle > EXTENED_FINGER_THRESHOLD:  # Threshold for extended finger
                extended_fingers.append(True)
            else:
                extended_fingers.append(False)


        #print(f"Extended fingers for {hand_label}: {extended_fingers}")
        return extended_fingers

    def _is_thumb_extended(self,hand_landmark, hand_label) -> bool:
        """
        Determine if the thumb is extended based on the cross product of vectors from  thumb MCP -> index MCP and  thumb MCP -> thumb tip. 
        The sign of the cross product indicates  if the thumb has crossed the thumb MCP -> index MCP line, and therefore if the thumb is extended or not, 
        taking into account handedness and palm orientation.
        """
        index_mcp = lm_to_vec(hand_landmark[HandLandmark.INDEX_FINGER_MCP]) 
        thumb_mcp = lm_to_vec(hand_landmark[HandLandmark.THUMB_MCP])
        thumb_tip = lm_to_vec(hand_landmark[HandLandmark.THUMB_TIP]) 

        cross = np.cross( index_mcp - thumb_mcp, thumb_tip - thumb_mcp)


        palm_facing = self._is_palm_facing(hand_landmark, hand_label)

        if cross == 0:
            return False  # Thumb is not extended if the cross product is zero (collinear)
        
        if hand_label == "Right":
            expected_sign = cross < 0
        else:
            expected_sign = cross > 0

        # Flip logic if palm is facing away
        if not palm_facing:
            expected_sign = not expected_sign
        result = bool(expected_sign)
        return result

    def _is_palm_facing(self, hand_landmark, hand_label) -> bool:
        """
        Infer palm direction from landmark winding order.
        Returns True if palm is facing camera.
        """
        mid    = lm_to_vec(hand_landmark[HandLandmark.MIDDLE_FINGER_MCP])
        wrist  = lm_to_vec(hand_landmark[HandLandmark.WRIST])
        pinky  = lm_to_vec(hand_landmark[HandLandmark.PINKY_MCP])
        cross = np.cross(mid - wrist, pinky - wrist)

        if cross == 0:
            return False  # Palm is not facing camera if the cross product is zero

        if hand_label == "Right":
            return cross > 0
        else:
            return cross < 0
        
    # --- Drawing Methods ---

    #assume is 1 hand
    def _draw_landmarks(self, frame, hand_landmark, hand_label, hand_score):
        """
        Draws the hand landmarks and connections on the frame.
        
        Args:
            frame: BGR image from cv2.VideoCapture to draw landmarks and connections.
            hand_landmark: List of MediaPipe hand landmarks for a single hand.
            hand_label: String label for the hand ("Left" or "Right").
            hand_score: Confidence score for the hand detection.
        """
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




    def _draw_angles(self, frame, hand_landmark, joint_list):
        """
        Draws the angle value at the middle joint of each triplet.
        Uses image coords (2D) for the math and draw position.
        hand_world_landmark and (3D) tested and dropped -- confirmed unreliaable in bug #5571
        """
        h, w = frame.shape[:2]
        for joint in joint_list:
            a, b, c = joint
            angle = angle_between(
                hand_landmark[a], 
                hand_landmark[b], 
                hand_landmark[c]
            )

            # Draw position of the middle joint in image coordinates
            pt = (int(hand_landmark[b].x * w), int(hand_landmark[b].y * h))
            cv2.putText(frame, f"{angle:.0f}", pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 2, cv2.LINE_AA)



    def _draw_extended_count(self,frame, num_extended:int) -> None:
        text = f"Count: {num_extended}"
        #print(num_extended)

        cv2.putText(frame, text, (250,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,0,200), cv2.LINE_AA)



    @staticmethod
    def _count_extended_fingers(extended_fingers: list) -> int:
        """
        Counts the total number of extended fingers across all hands.
        Args:
            extended_fingers: List or list of lists of bools indicating extended fingers for each hand.
        Returns:
            Total count of extended fingers.
        """
        if extended_fingers is None:
            return 0
        #Normailze to list of lists
        if not isinstance(extended_fingers[0], list):
            extended_fingers = [extended_fingers]
        num_extended = sum(1 for hand in extended_fingers for extended in hand if extended)
        return num_extended