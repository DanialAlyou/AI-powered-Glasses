import cv2
from ultralytics import YOLO
from gtts import gTTS
import pygame
from pydub import AudioSegment
from pydub.playback import play
import threading
import mediapipe as mp
from deepface import DeepFace
from collections import Counter
import time  # To add timestamp logging

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Load YOLO model with explicit task definition
model = YOLO('best_ncnn_model', task='detect')

# Classifications and system states
class_names = ["person", "bicycle", "car", "motorcycle", "bed", "bus", "desk", "table", "door", "fridge", "toilet",
               "sofa", "sink", "microwave", "bench", "chair", "fan", "closet", "stairs", "crosswalk",
               "green pedestrian Traffic Light, walk", "red pedestrian Traffic Light, stop", "face", "A", "B", "W", "Y"]
dangerous_classes = ['car', 'motorcycle', 'bus', 'red pedestrian Traffic Light, stop', 'stairs']
alert_classes = ["car", "bus", "bicycle", "motorcycle", "pedestrian Traffic Light (stop)"]
gesture_classes = ["A", "B", "W", "Y", "1", "2"]

# Global state variables
mode = 'indoor'  # Default mode
face_recognition = False
recognized_faces = {}  # Dictionary to store recognized faces (name: count)
is_audio_playing = False  # Flag to track audio playback
audio_lock = threading.Lock()  # Lock for safe threading
custom_command = False  # Default custom command state

# Initialize MediaPipe hands for gesture detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to generate and play audio
def generate_audio(text, dangerous=False, speed_factor=1.5):
    global is_audio_playing

    with audio_lock:
        if dangerous and is_audio_playing:
            pygame.mixer.music.stop()  # Interrupt current playback for danger
            is_audio_playing = False

        # Generate speech using gTTS
        tts = gTTS(text=text, lang='en')
        tts.save("output.mp3")

        # Load the audio file into pydub
        audio = AudioSegment.from_mp3("output.mp3")

        # Speed up the audio using the speed_factor
        fast_audio = audio.speedup(playback_speed=speed_factor)

        # Export the fast audio
        fast_audio.export("output_fast.mp3", format="mp3")

        # Get the duration of the sped-up audio (in seconds)
        audio_duration = (len(fast_audio) / 1000.0) - 1  # duration in seconds - 1
        # print(audio_duration)
        # Load the fast audio into pygame and play it
        pygame.mixer.music.load("output_fast.mp3")
        pygame.mixer.music.play()

        is_audio_playing = True

    # Sleep for the duration of the audio playback
    time.sleep(audio_duration)

    with audio_lock:
        is_audio_playing = False

    # Monitor audio playback in a separate thread (optional, can be removed)
    def monitor_audio():
        global is_audio_playing
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)  # Adding a small sleep to prevent hogging the CPU
        with audio_lock:
            is_audio_playing = False

    # Start the monitor_audio thread
    threading.Thread(target=monitor_audio, daemon=True).start()

# Function to split detections by sector and check for dangerous situations
def split_detections_by_sector(detections, width, height, class_names, alert_classes, mode=False):
    sectors = {
        "left": [0, int(width * 0.25)],
        "center": [int(width * 0.25), int(width * 0.75)],
        "right": [int(width * 0.75), width]
    }

    counts = {
        "left": {},
        "center": {},
        "right": {}
    }
    alert_status = {"alert_status": False, "object": None}

    if mode == "outdoor":
        outdoor_mode = True
    else:
        outdoor_mode = False

    for detection in detections[0].boxes.data:
        x1, y1, x2, y2, confidence, label_index = map(float, detection[:6])
        label = class_names[int(label_index)]

        center_x = (x1 + x2) / 2

        if center_x < sectors["left"][1]:
            sector = "left"
        elif center_x < sectors["center"][1]:
            sector = "center"
        else:
            sector = "right"

        if label in counts[sector]:
            counts[sector][label] += 1
        else:
            counts[sector][label] = 1

        # Check for alert in center sector if in outdoor mode
        if sector == "center" and outdoor_mode and label in alert_classes:
            alert_status = {"alert_status": True, "object": label}
            return alert_status

    return counts

# Function to generate text results
def generate_text_results(result, recognized_faces):
    messages = []
    if "alert_status" in result and result["alert_status"]:
        messages.append(f"Be careful, there is a {result['object']} in front of you")

    sectors_text = []
    sector_order = ["center", "left", "right"]
    sector_phrases = {
        "center": "in front of you",
        "left": "on your left",
        "right": "on your right"
    }

    for sector in sector_order:
        if sector in result and result[sector]:
            object_descriptions = []
            for label, count in result[sector].items():
                if label != "face":  # Don't include generic face count here
                    if count > 1:
                        object_descriptions.append(f"{count} {label}s")
                    else:
                        object_descriptions.append(f"{count} {label}")

            if object_descriptions:
                objects_text = ", and ".join(object_descriptions)
                sectors_text.append(f"there are {objects_text} {sector_phrases[sector]}")

    messages.extend(sectors_text)

    if recognized_faces:
        face_descriptions = []
        for name in recognized_faces:
            face_descriptions.append(f"{name}")
        if face_descriptions:
            messages.append(f"Recognized {', and '.join(face_descriptions)} in front of you.")

    return messages

# Functions for each gesture check
def check_gesture_A(landmarks):
    fingers_are_curled = (landmarks.landmark[8].y > landmarks.landmark[5].y and
                                landmarks.landmark[12].y > landmarks.landmark[9].y and
                                landmarks.landmark[16].y > landmarks.landmark[13].y and
                                landmarks.landmark[20].y > landmarks.landmark[17].y)

    if fingers_are_curled:
        return True
    return False

def check_gesture_B(landmarks):
    fingers_are_curled = (landmarks.landmark[8].y < landmarks.landmark[5].y and
                                landmarks.landmark[12].y < landmarks.landmark[9].y and
                                landmarks.landmark[16].y < landmarks.landmark[13].y and
                                landmarks.landmark[20].y < landmarks.landmark[17].y)
    if fingers_are_curled:
        return True
    return False

def check_gesture_W(landmarks):
    fingers_are_curled = (landmarks.landmark[8].y < landmarks.landmark[5].y and
                                landmarks.landmark[12].y < landmarks.landmark[9].y and
                                landmarks.landmark[16].y < landmarks.landmark[13].y and
                                landmarks.landmark[20].y > landmarks.landmark[17].y)

    if fingers_are_curled:
        return True
    return False

def check_gesture_Y(landmarks):
    fingers_are_curled = (landmarks.landmark[8].y > landmarks.landmark[5].y and
                                landmarks.landmark[12].y > landmarks.landmark[9].y and
                                landmarks.landmark[16].y > landmarks.landmark[13].y and
                                landmarks.landmark[20].y < landmarks.landmark[17].y)

    if fingers_are_curled:
        return True
    return False

def check_gesture_1(landmarks):
    fingers_are_curled = (landmarks.landmark[8].y < landmarks.landmark[5].y and
                                landmarks.landmark[12].y > landmarks.landmark[9].y and
                                landmarks.landmark[16].y > landmarks.landmark[13].y and
                                landmarks.landmark[20].y > landmarks.landmark[17].y)

    if fingers_are_curled:
        return True
    return False

def check_gesture_2(landmarks):
    fingers_are_curled = (landmarks.landmark[8].y < landmarks.landmark[5].y and
                                landmarks.landmark[12].y < landmarks.landmark[9].y and
                                landmarks.landmark[16].y > landmarks.landmark[13].y and
                                landmarks.landmark[20].y > landmarks.landmark[17].y)

    if fingers_are_curled:
        return True
    return False

# Function to detect hand gestures using MediaPipe
def detect_hand_gestures(frame):
    global mode, face_recognition, custom_command, is_audio_playing

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get the hand landmarks
    result = hands.process(rgb_frame)

    gestures = []

    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            # Check for gestures
            if check_gesture_A(landmarks):
                gestures.append("A")
                face_recognition = True
                # Generate audio feedback for gesture A
                if not is_audio_playing:
                    generate_audio("Face recognition enabled.", dangerous=False)

            elif check_gesture_B(landmarks):
                gestures.append("B")
                face_recognition = False
                # Generate audio feedback for gesture B
                if not is_audio_playing:
                    generate_audio("Face recognition disabled.", dangerous=False)

            elif check_gesture_W(landmarks):
                gestures.append("W")
                mode = "outdoor"
                # Generate audio feedback for gesture W
                if not is_audio_playing:
                    generate_audio("Mode set to outdoor.", dangerous=False)

            elif check_gesture_Y(landmarks):
                gestures.append("Y")
                mode = "indoor"
                # Generate audio feedback for gesture Y
                if not is_audio_playing:
                    generate_audio("Mode set to indoor.", dangerous=False)

            elif check_gesture_1(landmarks):
                gestures.append("1")
                custom_command = True
                # Generate audio feedback for gesture 1
                if not is_audio_playing:
                    generate_audio("Custom command enabled.", dangerous=False)

            elif check_gesture_2(landmarks):
                gestures.append("2")
                custom_command = False
                # Generate audio feedback for gesture 2
                if not is_audio_playing:
                    generate_audio("Custom command disabled.", dangerous=False)

    # Print detected gestures
    if gestures:
        print(f"Detected hand gestures: {', '.join(gestures)}")
    else:
        print("No hand gestures detected.")

    return gestures

# Function for face recognition using DeepFace
def recognize_faces(frame, database_path="DF_DB"):
    global recognized_faces
    recognized_faces = {}
    most_frequent_folder =[]
    # try:
    # Use DeepFace to find the closest matches from the database
    result = DeepFace.find(img_path=frame, model_name="Facenet", db_path=database_path, enforce_detection=False)

    # If results are found, process each match
    if result:
        for result_df in result:
            if 'identity' in result_df.columns:
                identities = result_df['identity'].values
                for identity in identities:
                    # Extract just the parent folder name (the most frequent one)
                    parent_folder_name = identity.split('/')[-2]  # Get the parent folder (folder name before the file)

                    # Count the occurrences of each folder name
                    if parent_folder_name in recognized_faces:
                        recognized_faces[parent_folder_name] += 1
                    else:
                        recognized_faces[parent_folder_name] = 1

                    print(Counter(recognized_faces).most_common(1))
            if recognized_faces:
                most_frequent_folder.append(Counter(recognized_faces).most_common(1)[0][0])
                
    # except Exception as e:
    #     print(f"Error during face recognition: {e}")
    
    # Return the most frequent folder name
    if recognized_faces:
        return most_frequent_folder
    else:
        return recognized_faces


def main():
    global mode, face_recognition, custom_command, is_audio_playing, recognized_faces

    show_camera = True  # Set to False if you don't want to display the camera frames

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_skip = 20  # Reduced frame skip for more frequent processing
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip == 0:
            # Show the live camera feed if enabled
            if show_camera:
                cv2.imshow("Camera Feed", frame)

            # Detect objects using YOLO
            detections = model(frame)

            # Perform face recognition if enabled
            if face_recognition:
                recognized_faces = recognize_faces(frame)
                print(f"Recognized faces: {recognized_faces}")
            else:
                recognized_faces = {}

            # Split detections by sector and check for dangerous situations
            result = split_detections_by_sector(detections, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                                class_names, alert_classes, mode == "outdoor")

            # Generate textual results including face recognition
            text_results = generate_text_results(result, recognized_faces)

            # Handle dangerous situations
            if "alert_status" in result and result["alert_status"]:
                generate_audio(f"Be careful, there is a {result['object']} in front of you", dangerous=True)
            elif not is_audio_playing and text_results:
                # Play audio for non-dangerous detections and recognized faces
                for text in text_results:
                    generate_audio(text)

            # Detect hand gestures
            gestures = detect_hand_gestures(frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# Main function
# def main():
#     global mode, face_recognition, custom_command, is_audio_playing, recognized_faces

#     # Initialize webcam
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     frame_skip = 20  # Reduced frame skip for more frequent processing
#     frame_count = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         if frame_count % frame_skip == 0:
#             # Show the live camera feed
#             cv2.imshow("Camera Feed", frame)

#             # Detect objects using YOLO
#             detections = model(frame)

#             # Perform face recognition if enabled
#             if face_recognition:
#                 recognized_faces = recognize_faces(frame)
#                 print(f"Recognized faces: {recognized_faces}")
#             else:
#                 recognized_faces = {}

#             # Split detections by sector and check for dangerous situations
#             result = split_detections_by_sector(detections, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
#                                                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
#                                                 class_names, alert_classes, mode == "outdoor")

#             # Generate textual results including face recognition
#             text_results = generate_text_results(result, recognized_faces)

#             # Handle dangerous situations
#             if "alert_status" in result and result["alert_status"]:
#                 generate_audio(f"Be careful, there is a {result['object']} in front of you", dangerous=True)
#             elif not is_audio_playing and text_results:
#                 # Play audio for non-dangerous detections and recognized faces
#                 for text in text_results:
#                     generate_audio(text)

#             # Detect hand gestures
#             gestures = detect_hand_gestures(frame)

#         # Press 'q' to exit the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


import cv2
from ultralytics import YOLO
from gtts import gTTS
import pygame
from pydub import AudioSegment
from pydub.playback import play
import threading
import mediapipe as mp
from deepface import DeepFace
from collections import Counter
import time  # To add timestamp logging
import paho.mqtt.client as mqtt

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Load YOLO model with explicit task definition
model = YOLO('best_ncnn_model', task='detect')

# MQTT Broker details
MQTT_BROKER = 'localhost'
MQTT_PORT = 1883
MQTT_TOPIC = '/leds/esp8266'

# Initialize MQTT client
mqtt_client = mqtt.Client()

# Classifications and system states
class_names = ["person", "bicycle", "car", "motorcycle", "bed", "bus", "desk", "table", "door", "fridge", "toilet",
                "sofa", "sink", "microwave", "bench", "chair", "fan", "closet", "stairs", "crosswalk",
                "green pedestrian Traffic Light, walk", "red pedestrian Traffic Light, stop", "face", "A", "B", "W", "Y"]
dangerous_classes = ['car', 'motorcycle', 'bus', 'red pedestrian Traffic Light, stop', 'stairs']
alert_classes = ["car", "bus", "bicycle", "motorcycle", "pedestrian Traffic Light (stop)"]
gesture_classes = ["A", "B", "W", "Y", "1", "2"]

# Global state variables
mode = 'indoor'  # Default mode
face_recognition = False
recognized_faces = {}  # Dictionary to store recognized faces (name: count)
is_audio_playing = False  # Flag to track audio playback
audio_lock = threading.Lock()  # Lock for safe threading
custom_command = False  # Default custom command state

# Initialize MediaPipe hands for gesture detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to generate and play audio
def generate_audio(text, dangerous=False, speed_factor=1.5):
    global is_audio_playing

    with audio_lock:
        if dangerous and is_audio_playing:
            pygame.mixer.music.stop()  # Interrupt current playback for danger
            is_audio_playing = False

        # Generate speech using gTTS
        tts = gTTS(text=text, lang='en')
        tts.save("output.mp3")

        # Load the audio file into pydub
        audio = AudioSegment.from_mp3("output.mp3")

        # Speed up the audio using the speed_factor
        fast_audio = audio.speedup(playback_speed=speed_factor)

        # Export the fast audio
        fast_audio.export("output_fast.mp3", format="mp3")

        # Get the duration of the sped-up audio (in seconds)
        audio_duration = len(fast_audio) / 1000.0  # duration in seconds

        # Load the fast audio into pygame and play it
        pygame.mixer.music.load("output_fast.mp3")
        pygame.mixer.music.play()

        is_audio_playing = True

    # Sleep for the duration of the audio playback
    time.sleep(audio_duration)

    with audio_lock:
        is_audio_playing = False

    # Monitor audio playback in a separate thread (optional, can be removed)
    def monitor_audio():
        global is_audio_playing
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)  # Adding a small sleep to prevent hogging the CPU
        with audio_lock:
            is_audio_playing = False

    # Start the monitor_audio thread
    threading.Thread(target=monitor_audio, daemon=True).start()

# Function to split detections by sector and check for dangerous situations
def split_detections_by_sector(detections, width, height, class_names, alert_classes, mode=False):
    sectors = {
        "left": [0, int(width * 0.25)],
        "center": [int(width * 0.25), int(width * 0.75)],
        "right": [int(width * 0.75), width]
    }

    counts = {
        "left": {},
        "center": {},
        "right": {}
    }
    alert_status = {"alert_status": False, "object": None}

    if mode == "outdoor":
        outdoor_mode = True
    else:
        outdoor_mode = False

    for detection in detections[0].boxes.data:
        x1, y1, x2, y2, confidence, label_index = map(float, detection[:6])
        label = class_names[int(label_index)]

        center_x = (x1 + x2) / 2

        if center_x < sectors["left"][1]:
            sector = "left"
        elif center_x < sectors["center"][1]:
            sector = "center"
        else:
            sector = "right"

        if label in counts[sector]:
            counts[sector][label] += 1
        else:
            counts[sector][label] = 1

        # Check for alert in center sector if in outdoor mode
        if sector == "center" and outdoor_mode and label in alert_classes:
            alert_status = {"alert_status": True, "object": label}
            return alert_status

    return counts

# Function to generate text results
def generate_text_results(result, recognized_faces):
    messages = []
    if "alert_status" in result and result["alert_status"]:
        messages.append(f"Be careful, there is a {result['object']} in front of you")

    sectors_text = []
    sector_order = ["center", "left", "right"]
    sector_phrases = {
        "center": "in front of you",
        "left": "on your left",
        "right": "on your right"
    }

    for sector in sector_order:
        if sector in result and result[sector]:
            object_descriptions = []
            for label, count in result[sector].items():
                if label != "face":  # Don't include generic face count here
                    if count > 1:
                        object_descriptions.append(f"{count} {label}s")
                    else:
                        object_descriptions.append(f"{count} {label}")

            if object_descriptions:
                objects_text = ", and ".join(object_descriptions)
                sectors_text.append(f"there are {objects_text} {sector_phrases[sector]}")

    messages.extend(sectors_text)

    if recognized_faces:
        face_descriptions = []
        for name in recognized_faces:
            face_descriptions.append(f"{name}")
        if face_descriptions:
            messages.append(f"Recognized {', and '.join(face_descriptions)} in front of you.")

    return messages

# Functions for each gesture check
def check_gesture_A(landmarks):
    fingers_are_curled = (landmarks.landmark[8].y > landmarks.landmark[5].y and
                            landmarks.landmark[12].y > landmarks.landmark[9].y and
                            landmarks.landmark[16].y > landmarks.landmark[13].y and
                            landmarks.landmark[20].y > landmarks.landmark[17].y)

    if fingers_are_curled:
        return True
    return False

def check_gesture_B(landmarks):
    fingers_are_curled = (landmarks.landmark[8].y < landmarks.landmark[5].y and
                            landmarks.landmark[12].y < landmarks.landmark[9].y and
                            landmarks.landmark[16].y < landmarks.landmark[13].y and
                            landmarks.landmark[20].y < landmarks.landmark[17].y)
    if fingers_are_curled:
        return True
    return False

def check_gesture_W(landmarks):
    fingers_are_curled = (landmarks.landmark[8].y < landmarks.landmark[5].y and
                            landmarks.landmark[12].y < landmarks.landmark[9].y and
                            landmarks.landmark[16].y < landmarks.landmark[13].y and
                            landmarks.landmark[20].y > landmarks.landmark[17].y)

    if fingers_are_curled:
        return True
    return False

def check_gesture_Y(landmarks):
    fingers_are_curled = (landmarks.landmark[8].y > landmarks.landmark[5].y and
                            landmarks.landmark[12].y > landmarks.landmark[9].y and
                            landmarks.landmark[16].y > landmarks.landmark[13].y and
                            landmarks.landmark[20].y < landmarks.landmark[17].y)

    if fingers_are_curled:
        return True
    return False

def check_gesture_1(landmarks):
    fingers_are_curled = (landmarks.landmark[8].y < landmarks.landmark[5].y and
                            landmarks.landmark[12].y > landmarks.landmark[9].y and
                            landmarks.landmark[16].y > landmarks.landmark[13].y and
                            landmarks.landmark[20].y > landmarks.landmark[17].y)

    if fingers_are_curled:
        return True
    return False

def check_gesture_2(landmarks):
    fingers_are_curled = (landmarks.landmark[8].y < landmarks.landmark[5].y and
                            landmarks.landmark[12].y < landmarks.landmark[9].y and
                            landmarks.landmark[16].y > landmarks.landmark[13].y and
                            landmarks.landmark[20].y > landmarks.landmark[17].y)

    if fingers_are_curled:
        return True
    return False

# Function to detect hand gestures using MediaPipe
def detect_hand_gestures(frame):
    global mode, face_recognition, custom_command, is_audio_playing

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get the hand landmarks
    result = hands.process(rgb_frame)

    gestures = []

    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            # Check for gestures
            if check_gesture_A(landmarks):
                gestures.append("A")
                face_recognition = True
                # Generate audio feedback for gesture A
                if not is_audio_playing:
                    generate_audio("Face recognition enabled.", dangerous=False)

            elif check_gesture_B(landmarks):
                gestures.append("B")
                face_recognition = False
                # Generate audio feedback for gesture B
                if not is_audio_playing:
                    generate_audio("Face recognition disabled.", dangerous=False)

            elif check_gesture_W(landmarks):
                gestures.append("W")
                mode = "outdoor"
                # Generate audio feedback for gesture W
                if not is_audio_playing:
                    generate_audio("Mode set to outdoor.", dangerous=False)

            elif check_gesture_Y(landmarks):
                gestures.append("Y")
                mode = "indoor"
                # Generate audio feedback for gesture Y
                if not is_audio_playing:
                    generate_audio("Mode set to indoor.", dangerous=False)

            elif check_gesture_1(landmarks):
                gestures.append("1")
                # Custom command ON
                if not is_audio_playing:
                    generate_audio("Custom command ON.", dangerous=False)
                return "ON" # Return the command

            elif check_gesture_2(landmarks):
                gestures.append("2")
                # Custom command OFF
                if not is_audio_playing:
                    generate_audio("Custom command OFF.", dangerous=False)
                return "OFF" # Return the command

    # Print detected gestures (excluding 1 and 2 as they trigger commands)
    filtered_gestures = [g for g in gestures if g not in ["1", "2"]]
    if filtered_gestures:
        print(f"Detected hand gestures: {', '.join(filtered_gestures)}")
    else:
        print("No relevant hand gestures detected.")

    return None # Return None if no custom command gesture is detected

# Function for face recognition using DeepFace
def recognize_faces(frame, database_path="DF_DB"):
    global recognized_faces
    recognized_faces = {}
    most_frequent_folder =[]
    # try:
    # Use DeepFace to find the closest matches from the database
    result = DeepFace.find(img_path=frame, model_name="Facenet", db_path=database_path, enforce_detection=False)

    # If results are found, process each match
    if result:
        for result_df in result:
            if 'identity' in result_df.columns:
                identities = result_df['identity'].values
                for identity in identities:
                    # Extract just the parent folder name (the most frequent one)
                    parent_folder_name = identity.split('/')[-2]  # Get the parent folder (folder name before the file)

                    # Count the occurrences of each folder name
                    if parent_folder_name in recognized_faces:
                        recognized_faces[parent_folder_name] += 1
                    else:
                        recognized_faces[parent_folder_name] = 1

                print(Counter(recognized_faces).most_common(1))
        if recognized_faces:
            most_frequent_folder.append(Counter(recognized_faces).most_common(1)[0][0])

    # except Exception as e:
    #    print(f"Error during face recognition: {e}")

    # Return the most frequent folder name
    if recognized_faces:
        return most_frequent_folder
    else:
        return recognized_faces

# Callback function for MQTT connection
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}")

# Connect to MQTT broker
mqtt_client.on_connect = on_connect
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start() # Start the MQTT client loop in the background

# Main function
def main():
    global mode, face_recognition, custom_command, is_audio_playing, recognized_faces

    show_cam = True
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_skip = 20  # Reduced frame skip for more frequent processing
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip == 0:
            # Show the live camera feed
            if show_cam:
                cv2.imshow("Camera Feed", frame)

            # Detect objects using YOLO
            detections = model(frame)

            # Perform face recognition if enabled
            if face_recognition:
                recognized_faces = recognize_faces(frame)
                print(f"Recognized faces: {recognized_faces}")
            else:
                recognized_faces = {}

            # Split detections by sector and check for dangerous situations
            result = split_detections_by_sector(detections, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                                class_names, alert_classes, mode == "outdoor")

            # Generate textual results including face recognition
            text_results = generate_text_results(result, recognized_faces)

            # Handle dangerous situations
            if "alert_status" in result and result["alert_status"]:
                generate_audio(f"Be careful, there is a {result['object']} in front of you", dangerous=True)
            elif not is_audio_playing and text_results:
                # Play audio for non-dangerous detections and recognized faces
                for text in text_results:
                    generate_audio(text)

            # Detect hand gestures and check for custom commands
            gesture_command = detect_hand_gestures(frame)
            if gesture_command == "ON":
                print("Sending ON command to NodeMCU via MQTT")
                mqtt_client.publish(MQTT_TOPIC, "ON")
            elif gesture_command == "OFF":
                print("Sending OFF command to NodeMCU via MQTT")
                mqtt_client.publish(MQTT_TOPIC, "OFF")

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    mqtt_client.loop_stop() # Stop the MQTT client loop

if __name__ ==