import cv2
from PIL import Image
import io
import numpy as np
import urllib.request
import os
import glob
import mediapipe as mp
import time
import pyautogui
import math
import webbrowser


def get_screenshot_from_frame(frame):
    """
    Converts a video frame represented as a NumPy array into a PIL Image.

    Parameters:
        frame (numpy.ndarray): The input video frame represented as a NumPy array, typically in BGR color format.

    Returns:
        PIL.Image.Image: A PIL Image object representing the input video frame in RGB color format.

    The function takes a single argument, 'frame', which is a NumPy array containing the pixel data of a video frame.
    It converts the BGR color format of the input frame to the RGB color format using OpenCV (cv2) functions
    and returns the resulting image as a PIL Image. This is useful for further image processing or display.
    """

    # Convert the frame to a PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return img

def save_screenshots(url, num_screenshots):
    """
    Capture and save screenshots from an online video feed to the local machine.

    Args:
        url (str): The URL of the online video feed to capture screenshots from.
        num_screenshots (int): The number of screenshots to capture and save.

    Returns:
        list of PIL.Image.Image: A list of PIL Image objects representing the captured screenshots.

    This function opens an online video feed specified by the 'url' parameter, captures a total of 'num_screenshots' screenshots
    from the feed, and saves each screenshot as a .jpg file on the local machine. A list of PIL Image objects is returned, 
    representing the captured screenshots. Filenames for the saved screenshots are generated based on a timestamp and a random 
    number for uniqueness.

    """

    # Open the video feed using urllib
    stream = urllib.request.urlopen(url)
    bytes = b''

    screenshots = []
    count = 0

    # Generate a unique filename based on timestamp and random number
    timestamp = int(time.time() * 1000)
    random_num = np.random.randint(0, 10000)

    # Take screenshot and save to local file
    while count < num_screenshots:
        bytes += stream.read(1024)
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes[a:b+2]
            bytes = bytes[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            screenshot = get_screenshot_from_frame(frame)
            filename = f"screenshot_{timestamp}_{random_num}_{count}.jpg"  # Update filename to include timestamp and random number
            screenshot.save(filename)
            screenshots.append(screenshot)
            count += 1

    # Close the video feed
    stream.close()

    return screenshots

def delete_all_screenshots(folder_path):
    """
    Capture screenshots from an online video feed, save them as .jpg files on the local machine, and return a list of captured screenshots.

    Args:
        url (str): The URL of the online video feed to capture screenshots from.
        num_screenshots (int): The number of screenshots to capture and save.

    Returns:
        list of PIL.Image.Image: A list of PIL Image objects, each representing a captured screenshot.

    This function accesses an online video feed specified by the 'url' parameter and captures 'num_screenshots' screenshots from the video feed. 
    It saves each screenshot as a .jpg file on the local machine, and also stores them as PIL Image objects in a list. 
    The filenames of the saved screenshots are generated based on the current timestamp and a random number to ensure uniqueness.
    """
    try:
        # Get the list of all files in the folder
        files = os.listdir(folder_path)

        # Loop through all files and delete image files
        for file in files:
            file_path = os.path.join(folder_path, file)
            if file.endswith(".jpg") or file.endswith(".png"):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")


    except Exception as e:
        print(f"Failed to delete image files: {e}")


def enhance_image(image):
    """
    Enhance the contrast of an input image using Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Args:
        image (numpy.ndarray): The input image to be enhanced.

    Returns:
        numpy.ndarray: The enhanced image with improved contrast.

    This function takes an input image and performs a contrast enhancement operation using Contrast Limited Adaptive Histogram Equalization (CLAHE). 
    It converts the input image to grayscale, applies CLAHE with specified parameters, and returns the enhanced image with improved contrast.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply contrast limited adaptive histogram equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)

    return enhanced


def are_fingers_outstretched(hand_landmarks):
    """
    Analyzes hand landmarks using the MediaPipe library to determine if the hand is in a 'high-five' gesture.

    Args:
        hand_landmarks (mediapipe.solutions.hands.HandLandmarks): Hand landmarks detected by the MediaPipe library.

    Returns:
        bool: True if the hand is in a 'high-five' gesture, False otherwise.

    This function takes a set of hand landmarks detected by the MediaPipe library and checks if the hand is in a 'high-five' gesture. 
    It determines the 'high-five' gesture based on the extension of the thumb and the fingers of the hand. The thumb is considered outstretched if its 
    tip is to the right of the metacarpophalangeal (MCP) joint. The other fingers (index, middle, ring, and pinky) are considered outstretched if their 
    tips are above their proximal interphalangeal (PIP) joints.
    """
    if hand_landmarks is None:
        return False

    # Thumb extended
    thumb_outstretched = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_MCP].x

    # All other fingers extended
    other_fingers_outstretched = all(
        [
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y <
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].y,
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y <
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].y,
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y <
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_PIP].y,
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP].y <
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_PIP].y
        ]
    )
    return thumb_outstretched and other_fingers_outstretched


def is_thumbs_up(hand_landmarks):
    """
    Analyze hand landmarks using the MediaPipe library to determine if the hand is making a 'thumbs-up' gesture.

    Args:
        hand_landmarks (mediapipe.solutions.hands.HandLandmarks): Hand landmarks detected by the MediaPipe library.

    Returns:
        bool: True if the hand is in a 'thumbs-up' gesture, False otherwise.

    This function takes a set of hand landmarks detected by the MediaPipe library and checks if the hand is making a 'thumbs-up' gesture. 
    It determines the 'thumbs-up' gesture based on the thumb pointing up and all other fingers folded into the palm. 
    Specifically, it checks if the vertical position of the thumb tip is above the thumb metacarpophalangeal (MCP) joint and if all other 
    fingers' tips are to the right of their proximal interphalangeal (PIP) joints.
    """
    if hand_landmarks is None:
        return False
    
    # Thumb pointed up
    thumb_outstretched = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_MCP].y

    # All other fingers folded into palm
    other_fingers_folded = all(
            [   hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x >
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].x,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].x >
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].x,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].x >
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_PIP].x,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP].x >
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_PIP].x
            ]
        )
        
    return thumb_outstretched and other_fingers_folded
    

def is_thumbs_down(hand_landmarks):
    """
    Analyze hand landmarks using the MediaPipe library to determine if the hand is making a 'thumbs-down' gesture.

    Args:
        hand_landmarks (mediapipe.solutions.hands.HandLandmarks): Hand landmarks detected by the MediaPipe library.

    Returns:
        bool: True if the hand is in a 'thumbs-down' gesture, False otherwise.

    This function takes a set of hand landmarks detected by the MediaPipe library and checks if the hand is making a 'thumbs-down' gesture. 
    It determines the 'thumbs-down' gesture based on the thumb pointing down and all other fingers folded into the palm. 
    Specifically, it checks if the vertical position of the thumb tip is below the thumb metacarpophalangeal (MCP) joint and if all other fingers' 
    tips are to the left of their proximal interphalangeal (PIP) joints.
    """
    if hand_landmarks is None:
        return False
    
    # Thumb pointed down
    thumb_folded = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y > hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_MCP].y
    
    # All other fingers folded into palm
    other_fingers_folded = all(
            [        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x <
                    hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].x,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].x <
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].x,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].x <
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_PIP].x,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP].x <
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_PIP].x
            ]
    )
        
    return thumb_folded and other_fingers_folded


def is_longhorn(hand_landmarks):
    """
    Analyze hand landmarks using the MediaPipe library to determine if the hand is making the 'Hook 'em' hand gesture.

    Args:
        hand_landmarks (mediapipe.solutions.hands.HandLandmarks): Hand landmarks detected by the MediaPipe library.

    Returns:
        bool: True if the hand is in the 'Hook 'em' gesture, False otherwise.

    This function takes a set of hand landmarks detected by the MediaPipe library and checks if the hand is making the 'Hook 'em' gesture. 
    The 'Hook 'em' gesture is characterized by specific finger positions:
    - Index and pinky fingers extended upward.
    - Middle and ring fingers folded into the palm.
    - The thumb can be either extended or folded.
    """
    if hand_landmarks is None:
        return False

    # Index and pinky fingers are extended
    index_tip_y = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y
    pinky_tip_y = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP].y
    fingers_extended = (index_tip_y < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP].y) and \
                       (pinky_tip_y < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP].y)

    # Middle two fingers are folded into the palm
    middle_tip_y = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip_y = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y
    fingers_folded = (middle_tip_y > hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP].y) and \
                     (ring_tip_y > hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_MCP].y)

    # Thumb can be extended or folded
    thumb_folded = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP].x
    thumb_extended = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y > hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP].y

    # Return True if all conditions are met
    return fingers_extended and fingers_folded and (thumb_folded or thumb_extended)


def main():
    """
    Analyze hand gestures using the MediaPipe HandPose model to perform various actions based on detected gestures.

    This main function performs the following tasks:
    1. Loads the MediaPipe HandPose model.
    2. Accesses an online server to capture a burst of screenshots.
    3. Enhances the selected screenshot using image processing.
    4. Analyzes hand landmarks in the enhanced screenshot.
    5. Performs specific actions based on detected hand gestures, including volume control, showing/hiding the desktop, and opening a web link.
    6. Cleans up by closing the MediaPipe model and deleting captured screenshots.
    """

    # Load the Mediapipe HandPose model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.01)

    # Online server URL
    #url = 'http://192.168.201.149'
    #url = 'http://192.168.222.149/'
    url = 'http://192.168.143.149'
    
    # Access online server and take burst of screenshots
    delete_all_screenshots(r"C:\Users\Evan_\Downloads\hand_images")
    num_screenshots = 7
    screenshots = save_screenshots(url, num_screenshots)

    # Select single screenshot for analyzation
    screenshot_to_analyze = screenshots[len(screenshots) // 2]
    frame = np.array(screenshot_to_analyze)
    
    # Apply image enhancement
    frame_enhanced = enhance_image(frame)
    
    # 
    result = hands.process(cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB))
    if result.multi_hand_landmarks:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        for hand_landmarks in result.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame_rgb, (x, y), 5, (0, 255, 0), -1)
            
            # If hand gesture is thumbs-up, increase volume
            if is_thumbs_up(hand_landmarks):
                print("Thumbs up detected!")
                for i in range(50):
                    pyautogui.press('volumeup')

            # If hand gesture is thumbs-down, lower volume
            elif is_thumbs_down(hand_landmarks):
                print("Thumbs down detected!")
                pyautogui.press('volumemute')            

            # If hand gesture is high-five, hide/show desktop    
            elif are_fingers_outstretched(hand_landmarks):
                print("All fingers outstretched detected!")
                pyautogui.hotkey('win', 'd')

            # If hand gesture is Hook 'em, play 'Never Gonna Give You Up' by Rick Astley
            elif is_longhorn(hand_landmarks):
                print("BEVO detected!")
                webbrowser.open_new_tab("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    
        #frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
        #cv2.imshow(f"Hand {1} with Landmarks", frame_bgr)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        hands.close()
        
        delete_all_screenshots(r'C:\Users\Evan_\Downloads\hand_images')

if __name__ == '__main__':
    # Run main function for checking hand gesture continuously 
    while True:
        main()
        time.sleep(1)
