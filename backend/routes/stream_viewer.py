import os
import cv2
import uuid
import clip
import json
import time
import torch
import joblib
import base64
import logging
import threading
import numpy as np
import queue
from PIL import Image
from queue import Queue
from collections import Counter
from firestore import body_drive_sessions, face_drive_sessions, obd_drive_sessions
from firebase_admin import firestore
from flask import Blueprint, Response, make_response, request
import dlib
from imutils import face_utils  # Helps convert dlib shapes to NumPy arrays

stream_viewer = Blueprint("stream_viewer", __name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Buffer for OBD data
obd_data_buffer = Queue(maxsize=1)

OBD_FIELDS = [
    'timestamp',
    'speed',
    'rpm',
    'check_engine_on',
    'num_dtc_codes',
    'dtc_codes',
    'warning_lights',
    # Optional warning light system fields
    'check_engine',
    'transmission',
    'abs',
    'airbag'
]

# Constants
# If your FACE_STREAM_URL and BODY_STREAM_URL are the same, you will get errors!
# FACE_STREAM_URL = "http://172.20.10.3/stream"  # ai thinker hotspot aaron
FACE_STREAM_URL = "http://192.168.0.105/stream"  # ai thinker home wifi aaron
# BODY_STREAM_URL = "http://172.20.10.8/stream"  # ai thinker hotspot aaron
BODY_STREAM_URL = "http://192.168.0.108/stream"  # wrover home wifi aaron
# Clip constants
CLIP_MODEL_NAME = "ViT-L/14"
CLIP_INPUT_SIZE = 224
CLIP_MODEL_PATH_BODY = "dmd29_vitbl14-hypc_429_1000_ft.pkl"
# Thresholds used for face stream
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.25

# Setup cv2 classifiers to detect a person in the frame
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
)
profileFaceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)
upperBodyCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_upperbody.xml"
)
# Setup cv2 classifier to detect eyes
leftEyeCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml"
)
rightEyeCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_righteye_2splits.xml"
)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, "models")

# Setup dlib face landmark predictor to find eyes
landmark_predictor = dlib.shape_predictor(
    os.path.join(model_dir, "shape_predictor_68_face_landmarks.dat")
)

# WHY BUFFER RATHER THAN JUST SAVING TO FIRESTORE IMMEDIATELY?
NUM_SECONDS_BEFORE_STORE_IN_DB = 30
frame_count_face = 0
face_frame_buffer_db = []
face_processing_thread = None
face_thread_kill = False
frame_count_body = 0
body_frame_buffer_db = []
body_processing_thread = None
body_thread_kill = False
obd_frame_buffer_db = []

# for timestamp of OBD, want to update on new server second only
last_processed_second = None
current_session_id = None

# THIS BUFFER IS FOR VIEWING THE STREAM AND PREDICTIONS LIVE
# small max length as we only want to show the most recent frames
# these queues must be thread safe as they are accessed by multiple threads
face_stream_data_buffer = Queue(maxsize=1)
body_stream_data_buffer = Queue(maxsize=1)

# These buffers will hold the predictions data for the current second
face_stream_cursecond_buffer = []
body_stream_cursecond_buffer = []

# Add global variables for clip model
clip_model = None
clip_preprocess = None
clip_classifier_body = None
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"CLIP Using device: {device}")

# TODO: Frontend code should also handle Unknown as a prediction
body_stream_index_to_label = {
    0: "Drinking beverage",
    1: "Adjusting hair, glasses, or makeup",
    2: "Talking on phone",
    3: "Reaching beside or behind",  # Was originally "Adjusting Radio or AC" but changed for performance
    4: "Reaching beside or behind",
    5: "Reaching beside or behind",
    6: "Driving Safely",
    7: "Talking to passenger",
    8: "Texting/using phone",
    9: "Yawning",
}


# Initialize CLIP components
def init_clip():
    global clip_model, clip_preprocess, clip_classifier_body
    clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device)
    clip_model.eval()
    clip_classifier_body = joblib.load(os.path.join(model_dir, CLIP_MODEL_PATH_BODY))


# Initialize CLIP components
try:
    init_clip()
    logger.info("CLIP model and classifier loaded successfully")
except Exception as e:
    logger.error(f"Error loading CLIP model: {str(e)}")
    raise


# Face stream helper function
# Each second classification will be stored in the format below
# face_drive_sessions/<sessionId>/face_drive_session_classifications/<timestamp>
# there will be a eye_classification and mouth_classification field with the final classification for that second
def save_face_frames_to_firestore(sessionId):
    """
    Saves classification data to Firestore under a document named after sessionId
    in the 'face_drive_sessions' collection. If the session doc does not exist, it is created.
    Otherwise, we update the existing doc with new timestamped classifications.
    """
    global face_frame_buffer_db

    if len(face_frame_buffer_db) >= NUM_SECONDS_BEFORE_STORE_IN_DB:
        logger.info("FACE_STREAM: Saving face frames to Firestore")

        # Copy the buffer so we don't mutate it while writing
        frame_data = list(face_frame_buffer_db)

        # 1) Create or retrieve the doc for this session ID
        doc_ref = face_drive_sessions.document(str(sessionId))
        doc_snapshot = doc_ref.get()

        if not doc_snapshot.exists:
            # If doc does not exist, create it
            doc_ref.set(
                {"session_id": str(sessionId), "created_at": firestore.SERVER_TIMESTAMP}
            )
            logger.info(
                f"FACE_STREAM: Created new session doc for session ID {sessionId}"
            )

        # 2) Write each (timestamp, classification) as a doc in 'face_drive_session_classifications'
        db_start_time = time.time()

        for record in frame_data:
            ts = record["timestamp"]  # integer second or unique timestamp
            eye_label = record["eye_classification"]  # eye classification
            mouth_label = record["mouth_classification"]  # eye classification

            # Document ID = timestamp; store both timestamp and classification
            doc_ref.collection("face_drive_session_classifications").document(
                str(ts)
            ).set(
                {
                    "timestamp": ts,
                    "eye_classification": eye_label,
                    "mouth_classification": mouth_label,
                }
            )

        db_time = (time.time() - db_start_time) * 1000  # milliseconds
        logger.info(f"FACE_STREAM: Database save completed in {db_time:.2f}ms")

        # 3) Clear the local buffer and reset
        face_frame_buffer_db = []


# Body stream helper function
# Each second classification will be stored in the format below
# body_drive_sessions/<sessionId>/body_drive_session_classifications/<timestamp>
# there will be a single classification field with the final classification for that second
def save_body_frames_to_firestore(sessionId):
    """
    Saves classification data to Firestore under a document named after sessionId
    in the 'body_drive_sessions' collection. If the session doc does not exist, it is created.
    Otherwise, we update the existing doc with new timestamped classifications.
    """
    global body_frame_buffer_db

    if len(body_frame_buffer_db) >= NUM_SECONDS_BEFORE_STORE_IN_DB:
        logger.info("BODY_STREAM: Saving body frames to Firestore")

        # Copy the buffer so we don't mutate it while writing
        frame_data = list(body_frame_buffer_db)

        # 1) Create or retrieve the doc for this session ID
        doc_ref = body_drive_sessions.document(str(sessionId))
        doc_snapshot = doc_ref.get()

        if not doc_snapshot.exists:
            # If doc does not exist, create it
            doc_ref.set(
                {"session_id": str(sessionId), "created_at": firestore.SERVER_TIMESTAMP}
            )
            logger.info(
                f"BODY_STREAM: Created new session doc for session ID {sessionId}"
            )

        # 2) Write each (timestamp, classification) as a doc in 'body_drive_session_classifications'
        db_start_time = time.time()

        for record in frame_data:
            ts = record["timestamp"]  # integer second or unique timestamp
            label = record["classification"]  # your final classification

            # Document ID = timestamp; store both timestamp and classification
            doc_ref.collection("body_drive_session_classifications").document(
                str(ts)
            ).set({"timestamp": ts, "classification": label})

        db_time = (time.time() - db_start_time) * 1000  # milliseconds
        logger.info(f"BODY_STREAM: Database save completed in {db_time:.2f}ms")

        # 3) Clear the local buffer and reset
        body_frame_buffer_db = []


# Function to help send OBD data to firestore database
# Each second's OBD frame data will be stored in the format below
# obd_drive_sessions/<sessionId>/obd_drive_session_classifications/<timestamp>
def save_obd_frames_to_firestore(sessionId):
    """
    Saves OBD data to Firestore under a document named after sessionId
    in the 'obd_drive_sessions' collection. Each document contains all available OBD fields.
    """
    
    global obd_frame_buffer_db
    
    if len(obd_frame_buffer_db) >= NUM_SECONDS_BEFORE_STORE_IN_DB:
        logger.info("OBD_STREAM: Saving OBD frames to Firestore")
        
        # Copy the buffer so we don't mutate it while writing
        frame_data = list(obd_frame_buffer_db)
        
        # 1) Create or retrieve the doc for this session ID
        doc_ref = obd_drive_sessions.document(str(sessionId))
        doc_snapshot = doc_ref.get()
        
        if not doc_snapshot.exists:
            # If doc does not exist, create it
            doc_ref.set(
                {"session_id": str(sessionId), "created_at": firestore.SERVER_TIMESTAMP}
            )
            logger.info(
                f"OBD_STREAM: Created new session doc for session ID {sessionId}"
            )
            
        # 2) Write each entry as a doc in 'obd_drive_session_classifications'
        db_start_time = time.time()

        for record in frame_data:
            ts = record["timestamp"]  # integer second or unique timestamp
            
            doc_data = {
                "timestamp": ts, 
                "speed": record.get("speed", -1),
                "rpm": record.get("rpm", -1),
                "check_engine_on": record.get("check_engine_on", False),
                "num_dtc_codes": record.get("num_dtc_codes", 0)
            }
            
            if "dtc_codes" in record:
                doc_data["dtc_codes"] = record["dtc_codes"]
                
            if "warning_lights" in record:
                doc_data["warning_lights"] = record["warning_lights"]

            # Document ID = timestamp;
            doc_ref.collection("obd_drive_session_classifications").document(str(ts)).set(doc_data)

        db_time = (time.time() - db_start_time) * 1000  # milliseconds
        logger.info(f"OBD_STREAM: Database save completed in {db_time:.2f}ms")
        
        # 3) Clear the local buffer and reset
        obd_frame_buffer_db = []

def process_obd_data(data, sessionId):
    """
    Process incoming OBD data and store it in the buffer for the current second.
    When the second changes, finalize the previous second's data.
    """
    global last_processed_second, obd_frame_buffer_db

    # Get the current second from the timestamp
    current_second = int(data["timestamp"])

    # If this is a new second, process the previous second's data
    if last_processed_second is not None and current_second != last_processed_second:
        # Create a record with all available OBD fields
        record = {
            "timestamp": last_processed_second,
            "speed": data.get("speed", -1),
            "rpm": data.get("rpm", -1),
            "check_engine_on": data.get("check_engine_on", False),
            "num_dtc_codes": data.get("num_dtc_codes", 0)
        }
        
        if "dtc_codes" in data:
            record["dtc_codes"] = data["dtc_codes"]
            
        if "warning_lights" in data:
            record["warning_lights"] = data["warning_lights"]
        
        # Add all available fields from the data
        # for field in OBD_FIELDS:
        #     if field in data and data[field] is not None:
        #         record[field] = data[field]

        # Add to buffer and possibly trigger a database write
        if len(obd_frame_buffer_db) < NUM_SECONDS_BEFORE_STORE_IN_DB:
            obd_frame_buffer_db.append(record)
        if len(obd_frame_buffer_db) >= NUM_SECONDS_BEFORE_STORE_IN_DB:
            save_obd_frames_to_firestore(sessionId)

    # Update the last processed second
    last_processed_second = current_second

# Face stream helper function to compute Eye Aspect Ratio (EAR)
# Formula taken from https://pmc.ncbi.nlm.nih.gov/articles/PMC11398282/
def compute_EAR(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Face stream helper function to compute Mouth Aspect Ratio (MAR)
# Formula taken from https://pmc.ncbi.nlm.nih.gov/articles/PMC11398282/
def compute_MAR(mouth):
    A = np.linalg.norm(mouth[1] - mouth[7])
    B = np.linalg.norm(mouth[2] - mouth[6])
    C = np.linalg.norm(mouth[3] - mouth[4])
    D = np.linalg.norm(mouth[0] - mouth[4])
    mar = (A + B + C) / (2.0 * D)
    return mar


# Eyes State Predictions are Eyes Open, Eyes Closed or Unknown
# Mouth State Predictions are Mouth Open, Mouth Closed or Unknown
# Used paper here https://pmc.ncbi.nlm.nih.gov/articles/PMC11398282/
def process_stream_face(url, sessionId):
    global frame_count_face, face_stream_data_buffer, face_thread_kill
    global face_frame_buffer_db, face_stream_cursecond_buffer

    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    from collections import Counter

    # Track "last second" for per-second majority
    last_second = None

    while not face_thread_kill:
        pipeline_start_time = time.time()

        # Initialize timing variables
        frame_capture_time = 0
        face_detect_time = 0
        dlib_time = 0
        ear_time = 0
        mar_time = 0
        encoding_time = 0
        db_time = 0

        # 1) Determine the current integer second
        now_second = int(time.time())

        # 2) If we have moved to a new second, finalize the previous second
        if last_second is not None and now_second != last_second:
            if len(face_stream_cursecond_buffer) > 0:
                # We have accumulated eye/mouth classifications for the old second in face_stream_cursecond_buffer
                eye_labels = [item["eye"] for item in face_stream_cursecond_buffer]
                mouth_labels = [item["mouth"] for item in face_stream_cursecond_buffer]

                # Majority vote for eye
                eye_counts = Counter(eye_labels)
                majority_eye, _ = eye_counts.most_common(1)[0]

                # Majority vote for mouth
                mouth_counts = Counter(mouth_labels)
                majority_mouth, _ = mouth_counts.most_common(1)[0]

                # Build a record for DB
                record = {
                    "timestamp": last_second,  # integer second
                    "eye_classification": majority_eye,
                    "mouth_classification": majority_mouth,
                }

                # Store to the DB buffer
                if len(face_frame_buffer_db) < NUM_SECONDS_BEFORE_STORE_IN_DB:
                    face_frame_buffer_db.append(record)
                if len(face_frame_buffer_db) >= NUM_SECONDS_BEFORE_STORE_IN_DB:
                    save_face_frames_to_firestore(sessionId)

                # Optionally push a “final” entry to the front-end with the aggregated classification
                # and a representative image. We do the same approach as the body stream: pop one item
                # from the queue if it is full, so we can reuse that item’s image.
                current_entry = None
                if face_stream_data_buffer.full():
                    current_entry = face_stream_data_buffer.get_nowait()

                # If we got something, use its image; otherwise None
                final_image = current_entry.get("image") if current_entry else None

                face_stream_data_buffer.put(
                    {
                        "image": final_image,
                        "eye_prediction": majority_eye,
                        "mouth_prediction": majority_mouth,
                        "ear_score": None,  # or best guess from aggregator if you like
                        "mar_score": None,
                        "frame_number": None,
                        "timestamp": last_second,
                        "processing_time": None,
                    }
                )

                # Clear the buffer for the old second
                face_stream_cursecond_buffer.clear()

        # 3) Update last_second
        if last_second is None:
            last_second = now_second
        elif now_second != last_second:
            last_second = now_second

        # 4) Capture frame
        capture_start = time.time()
        success, frame = cap.read()
        frame_capture_time = (time.time() - capture_start) * 1000

        if not success:
            logger.error(f"FACE_STREAM: Failed to read frame from {url}")
            time.sleep(0.1)
            continue

        frame_count_face += 1

        # Default
        eye_classification = "Unknown"
        mouth_classification = "Unknown"
        ear_score = None
        mar_score = None

        # 5) Detect face(s)
        face_detect_start = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 1)
        face_detect_time = (time.time() - face_detect_start) * 1000

        frame_with_boxes = frame.copy()

        # If no face, we draw "Unknown"
        if len(faces) == 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            text_to_draw = "Unknown"
            (text_width, text_height), baseline = cv2.getTextSize(
                text_to_draw, font, font_scale, thickness
            )
            padding = 10
            text_x = frame_with_boxes.shape[1] - text_width - padding
            text_y = text_height + padding
            cv2.putText(
                frame_with_boxes,
                text_to_draw,
                (text_x, text_y),
                font,
                font_scale,
                (0, 165, 255),  # Orange
                thickness,
            )
        else:
            # Find largest face
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            fx, fy, fw, fh = largest_face
            cv2.rectangle(
                frame_with_boxes, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2
            )

            # 6) Landmark detection (eyes, mouth, etc.)
            dlib_start = time.time()
            dlib_rect = dlib.rectangle(int(fx), int(fy), int(fx + fw), int(fy + fh))
            shape = landmark_predictor(gray, dlib_rect)
            shape_np = face_utils.shape_to_np(shape)
            dlib_time = (time.time() - dlib_start) * 1000

            # Eye Aspect Ratio
            ear_start = time.time()
            left_eye_pts = shape_np[36:42]
            right_eye_pts = shape_np[42:48]
            for x, y in np.concatenate((left_eye_pts, right_eye_pts)):
                cv2.circle(frame_with_boxes, (x, y), 1, (255, 255, 0), -1)
            left_ear = compute_EAR(left_eye_pts)
            right_ear = compute_EAR(right_eye_pts)
            avg_ear = (left_ear + right_ear) / 2.0
            ear_score = avg_ear
            eye_classification = (
                "Eyes Open" if avg_ear > EAR_THRESHOLD else "Eyes Closed"
            )
            ear_time = (time.time() - ear_start) * 1000

            # Draw bounding box around eyes
            eyes_pts = np.concatenate((left_eye_pts, right_eye_pts))
            (ex, ey, ew, eh) = cv2.boundingRect(eyes_pts)
            eye_color = (
                (0, 255, 0) if eye_classification == "Eyes Open" else (0, 0, 255)
            )
            cv2.rectangle(frame_with_boxes, (ex, ey), (ex + ew, ey + eh), eye_color, 2)
            eye_text = f"{eye_classification} ({ear_score:.2f})"
            cv2.putText(
                frame_with_boxes,
                eye_text,
                (ex, ey - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                eye_color,
                2,
            )

            # Mouth Aspect Ratio
            mar_start = time.time()
            mouth_pts = shape_np[60:68]
            mar_score = compute_MAR(mouth_pts)
            mouth_classification = (
                "Mouth Open" if mar_score > MAR_THRESHOLD else "Mouth Closed"
            )
            mar_time = (time.time() - mar_start) * 1000

            # Draw mouth landmarks
            for mx, my in mouth_pts:
                cv2.circle(frame_with_boxes, (mx, my), 1, (255, 255, 0), -1)
            (mx, my, mw, mh) = cv2.boundingRect(mouth_pts)
            mouth_color = (
                (0, 255, 0) if mouth_classification == "Mouth Closed" else (0, 0, 255)
            )
            cv2.rectangle(
                frame_with_boxes, (mx, my), (mx + mw, my + mh), mouth_color, 2
            )
            mouth_text = f"{mouth_classification} ({mar_score:.2f})"
            cv2.putText(
                frame_with_boxes,
                mouth_text,
                (mx, my - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                mouth_color,
                2,
            )

        # 7) Encode frame for streaming
        encode_start = time.time()
        _, buffer = cv2.imencode(".jpg", frame_with_boxes)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")
        encoding_time = (time.time() - encode_start) * 1000

        # 8) Store the immediate (eye, mouth) classification for majority voting
        #    but do NOT push the final classification to the front-end yet.
        face_stream_cursecond_buffer.append(
            {
                "eye": eye_classification,
                "mouth": mouth_classification,
            }
        )

        # 9) Put partial entry to the queue with classification = None
        #    so the front-end sees the live image, but no final classification.
        if face_stream_data_buffer.full():
            face_stream_data_buffer.get_nowait()

        total_time = (time.time() - pipeline_start_time) * 1000

        face_stream_data_buffer.put(
            {
                "image": frame_base64,
                "eye_prediction": None,  # hide classification until second ends
                "mouth_prediction": None,  # hide classification
                "ear_score": ear_score,  # optional to show or hide
                "mar_score": mar_score,
                "frame_number": frame_count_face,
                "timestamp": now_second,
                "processing_time": total_time,
            }
        )

        # Log timing
        # (We've removed the direct DB logic from each frame,
        #  as we only store once per second in the aggregator approach.)
        timing_log = (
            f"FACE_STREAM: Frame {frame_count_face} pipeline:\n"
            f"  - Frame Capture: {frame_capture_time:.2f}ms\n"
            f"  - Face Detection: {face_detect_time:.2f}ms\n"
        )
        if len(faces) > 0:
            fw, fh = largest_face[2], largest_face[3]
            timing_log += (
                f"  - Dlib Landmarks: {dlib_time:.2f}ms\n"
                f"  - EAR Calculation: {ear_time:.2f}ms\n"
                f"  - MAR Calculation: {mar_time:.2f}ms\n"
                f"  - Face Size: {fw * fh} px\n"
                f"  - EAR Score: {ear_score:.3f}\n"
                f"  - MAR Score: {mar_score:.3f}\n"
                f"  - Eye Classification: {eye_classification}\n"
                f"  - Mouth Classification: {mouth_classification}\n"
            )
        timing_log += (
            f"  - Frame Encoding: {encoding_time:.2f}ms\n"
            f"  - DB Time: {db_time:.2f}ms\n"
            f"  - Total Pipeline: {total_time:.2f}ms\n"
        )
        logger.info(timing_log)


# Body stream helper function
def preprocess_frame_clip(frame, preprocess):
    # Convert BGR to RGB and to PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    # Apply CLIP preprocessing
    processed = preprocess(pil_image)
    # Add batch dimension
    return processed.unsqueeze(0)


# Body stream helper function
# Detect a person in frame so we predict Unknown if there is no person detected rather
# than confidently predicting safe driving when there is noone in the frame
# This needs to be extremely fast as it will be called every frame
def detect_person_in_frame(frame, scale_factor=1.2, min_neighbors=1):
    start_time = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Body stream order: profile face -> upper body -> frontal face
    profile_faces = profileFaceCascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors
    )
    if len(profile_faces) > 0:
        detection_time = (time.time() - start_time) * 1000
        logger.info(
            f"BODY_STREAM: Person detection completed in {detection_time:.2f}ms (profile face detected)"
        )
        return True

    upper_bodies = upperBodyCascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors
    )
    if len(upper_bodies) > 0:
        detection_time = (time.time() - start_time) * 1000
        logger.info(
            f"BODY_STREAM: Person detection completed in {detection_time:.2f}ms (upper body detected)"
        )
        return True

    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors
    )
    detection_time = (time.time() - start_time) * 1000
    logger.info(
        f"BODY_STREAM: Person detection completed in {detection_time:.2f}ms (Person {'detected' if len(faces) > 0 else 'not detected'})"
    )
    return len(faces) > 0


# Expect body_stream_index_to_label classifications and Unknown as predictions on each frame
def process_stream_body(url, sessionId):
    global frame_count_body, clip_model, clip_preprocess, clip_classifier_body
    global body_stream_data_buffer, body_frame_buffer_db, body_thread_kill, body_stream_cursecond_buffer

    if not all([clip_model, clip_preprocess, clip_classifier_body]):
        logger.error("BODY_STREAM: CLIP components not initialized")
        return

    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # --- NEW CODE ---
    # Track the "last second" so we know when a new second starts.
    # We will accumulate predictions in `body_stream_cursecond_buffer` for that second.

    last_second = None

    while not body_thread_kill:
        pipeline_start_time = time.time()
        frame_capture_time = 0
        preprocess_time = 0
        feature_extraction_time = 0
        prediction_time = 0
        encoding_time = 0
        db_time = 0

        # --- NEW CODE ---
        # Figure out which second we are in right now.
        now_second = int(time.time())

        # If we've just moved to a new second, finalize the previous second's predictions.
        if last_second is not None and now_second != last_second:
            # If we have any buffered predictions from the last second, pick the majority.
            if len(body_stream_cursecond_buffer) > 0:
                counts = Counter(body_stream_cursecond_buffer)
                majority_label, _ = counts.most_common(1)[0]

                # 1) Build a dict containing both timestamp and classification
                record = {
                    "timestamp": last_second,  # the integer second
                    "classification": majority_label,
                }

                # 2) Save it in the local DB buffer
                if len(body_frame_buffer_db) < NUM_SECONDS_BEFORE_STORE_IN_DB:
                    body_frame_buffer_db.append(record)
                if len(body_frame_buffer_db) >= NUM_SECONDS_BEFORE_STORE_IN_DB:
                    save_body_frames_to_firestore(sessionId)

                # 3) Optionally also push a “final” entry to the front-end
                current_entry = None
                if body_stream_data_buffer.full():
                    current_entry = body_stream_data_buffer.get_nowait()
                # We are contending with the stream viewer code to get this entry in the queue
                # If the stream viewer code gets it first, that's fine, we'll lose the image and
                # we can just set the image to None
                image_from_entry = (
                    current_entry.get("image", None) if current_entry else None
                )
                body_stream_data_buffer.put(
                    {
                        "image": image_from_entry,
                        "prediction": majority_label,
                        "probability": None,
                        "frame_number": None,
                        "timestamp": last_second,
                        "processing_time": None,
                    }
                )

                # Clear the current-second buffer
                body_stream_cursecond_buffer.clear()

        # Update last_second if needed. First time through it might be None.
        if last_second is None:
            last_second = now_second
        elif now_second != last_second:
            # We finalized above, so now just set last_second
            last_second = now_second

        # ------------------------------------------------------------------
        # -- ORIGINAL CODE BELOW (with slight modifications for new logic) --
        # ------------------------------------------------------------------

        # Time frame capture
        capture_start = time.time()
        success, frame = cap.read()
        frame_capture_time = (time.time() - capture_start) * 1000

        if not success:
            logger.error(f"BODY_STREAM: Failed to read frame from {url}")
            time.sleep(1.0)
            continue

        frame_count_body += 1
        prediction = None
        prob_score = None
        prediction_label = "Unknown"

        # Create a copy for visualization
        frame_with_text = frame.copy()

        person_detected = True  # Not using person detection for now

        if person_detected:
            try:
                # Time preprocessing
                preprocess_start = time.time()
                processed = preprocess_frame_clip(frame, clip_preprocess).to(device)
                preprocess_time = (time.time() - preprocess_start) * 1000

                # Time feature extraction
                feature_start = time.time()
                with torch.no_grad():
                    features = clip_model.encode_image(processed)
                    features = features.cpu().numpy()
                feature_extraction_time = (time.time() - feature_start) * 1000

                # Time prediction
                # Using linear regression model trained here for prediction based on clip feature output
                # https://github.com/zahid-isu/DriveCLIP/tree/main?tab=readme-ov-file
                prediction_start = time.time()
                prediction = int(clip_classifier_body.predict(features)[0])
                prob_score = clip_classifier_body.predict_proba(features)[0][prediction]
                prediction_label = body_stream_index_to_label.get(prediction, "Unknown")
                prediction_time = (time.time() - prediction_start) * 1000

                # Set color based on prediction type
                if prediction_label == "Driving Safely":
                    text_color = (0, 255, 0)  # Green
                elif prediction_label == "Unknown":
                    text_color = (0, 165, 255)  # Orange
                else:
                    text_color = (0, 0, 255)  # Red for unsafe behaviors

                # Draw prediction with probability in top right
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                text_to_draw = f"{prediction_label} ({prob_score:.2f})"

                (text_width, text_height), baseline = cv2.getTextSize(
                    text_to_draw, font, font_scale, thickness
                )

                padding = 10
                text_x = frame_with_text.shape[1] - text_width - padding
                text_y = text_height + padding

                cv2.putText(
                    frame_with_text,
                    text_to_draw,
                    (text_x, text_y),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                )

                # Time frame encoding
                encode_start = time.time()
                _, buffer = cv2.imencode(".jpg", frame_with_text)
                frame_base64 = base64.b64encode(buffer).decode("utf-8")
                encoding_time = (time.time() - encode_start) * 1000

                # ----------------------------------------------------------------
                # NOTE: The database logic for per-frame storage has been moved
                #       to the "once a second has passed" block above, so we remove
                #       the direct DB-append code here.
                # ----------------------------------------------------------------

                # Calculate total pipeline time and log all metrics
                total_time = (time.time() - pipeline_start_time) * 1000

                timing_log = (
                    f"BODY_STREAM: Frame {frame_count_body} complete pipeline breakdown:\n"
                    f"  - Frame Capture: {frame_capture_time:.2f}ms\n"
                    f"  - Preprocessing: {preprocess_time:.2f}ms\n"
                    f"  - Feature Extraction: {feature_extraction_time:.2f}ms\n"
                    f"  - Model Prediction: {prediction_time:.2f}ms\n"
                    f"  - Frame Encoding: {encoding_time:.2f}ms\n"
                    f"  - Database Operations: {db_time:.2f}ms\n"
                    f"  - Total Pipeline Time: {total_time:.2f}ms\n"
                    f"  - Prediction: {prediction_label}\n"
                    f"  - Probability: {prob_score:.3f}"
                )
                logger.info(timing_log)

            except Exception as e:
                logger.error(f"BODY_STREAM: Error in inference: {str(e)}")
                prediction_label = "Unknown"
                prob_score = None

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                text_to_draw = "Unknown"

                (text_width, text_height), baseline = cv2.getTextSize(
                    text_to_draw, font, font_scale, thickness
                )

                padding = 10
                text_x = frame_with_text.shape[1] - text_width - padding
                text_y = text_height + padding

                cv2.putText(
                    frame_with_text,
                    text_to_draw,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (0, 165, 255),  # Orange in BGR
                    thickness,
                )

                _, buffer = cv2.imencode(".jpg", frame_with_text)
                frame_base64 = base64.b64encode(buffer).decode("utf-8")
        else:
            # Draw "Unknown" for no person detected with consistent positioning
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            text_to_draw = "Unknown"

            (text_width, text_height), baseline = cv2.getTextSize(
                text_to_draw, font, font_scale, thickness
            )

            padding = 10
            text_x = frame_with_text.shape[1] - text_width - padding
            text_y = text_height + padding

            cv2.putText(
                frame_with_text,
                text_to_draw,
                (text_x, text_y),
                font,
                font_scale,
                (0, 165, 255),  # Orange in BGR
                thickness,
            )

            encode_start = time.time()
            _, buffer = cv2.imencode(".jpg", frame_with_text)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")
            encoding_time = (time.time() - encode_start) * 1000

            total_time = (time.time() - pipeline_start_time) * 1000
            logger.info(
                f"BODY_STREAM: Frame {frame_count_body} pipeline breakdown (no person):\n"
                f"  - Frame Capture: {frame_capture_time:.2f}ms\n"
                f"  - Frame Encoding: {encoding_time:.2f}ms\n"
                f"  - Total Time: {total_time:.2f}ms\n"
                f"  - Prediction: Unknown (No person detected)"
            )

        # --- NEW CODE ---
        # Regardless of how we got the label (or unknown), we add it to our
        # body_stream_cursecond_buffer for the current second so we can do a
        # majority vote later.
        body_stream_cursecond_buffer.append(prediction_label)

        # Also, while we are still in this second, we do NOT show the prediction
        # on the frontend yet, so set 'prediction': None in body_stream_data_buffer.
        # (If the buffer is full, pop one to make room.)
        if body_stream_data_buffer.full():
            body_stream_data_buffer.get_nowait()

        total_time = (time.time() - pipeline_start_time) * 1000

        body_stream_data_buffer.put(
            {
                "image": frame_base64,
                "prediction": None,  # <--- Hide actual prediction for now
                "probability": None,  # <--- Also hide probability
                "frame_number": frame_count_body,
                "timestamp": now_second,  # integer second
                "processing_time": total_time,
            }
        )


# PUT ALL ROUTES BELOW -----------------------------------------------------------------------


# Start both streams as part of the same session
# Important because we need both streams to have the same session ID
@stream_viewer.route("/both_streams_start")
def both_streams_start():
    global body_processing_thread, face_processing_thread, current_session_id

    if body_processing_thread and body_processing_thread.is_alive():
        return make_response("Body processing thread already running", 409)

    if face_processing_thread and face_processing_thread.is_alive():
        return make_response("Face processing thread already running", 409)

    sessionId = uuid.uuid4()
    current_session_id = sessionId
    body_processing_thread = threading.Thread(
        target=process_stream_body, args=(BODY_STREAM_URL, sessionId)
    )
    body_processing_thread.start()

    face_processing_thread = threading.Thread(
        target=process_stream_face,
        args=(FACE_STREAM_URL, sessionId),
    )
    face_processing_thread.start()
    return make_response(
        f"Body and Face processing started for session {sessionId}", 200
    )


# Attempt to stop both streams
@stream_viewer.route("/both_streams_stop")
def both_streams_stop():
    global body_processing_thread, body_thread_kill, face_processing_thread, face_thread_kill

    errors = []
    success_messages = []

    # Stop body processing thread
    try:
        if body_processing_thread:
            if body_processing_thread.is_alive():
                body_thread_kill = True
                body_processing_thread.join(timeout=5.0)

                if body_processing_thread.is_alive():
                    errors.append("Failed to stop body thread within timeout")
                else:
                    success_messages.append("Body processing stopped successfully")
                    body_thread_kill = False
                    body_processing_thread = None
            else:
                body_processing_thread = None
                success_messages.append("Body thread was already stopped")
        else:
            success_messages.append("Body thread not started")
    except Exception as e:
        errors.append(f"Error stopping body thread: {str(e)}")
        logger.error(f"Error stopping body processing thread: {str(e)}")

    # Stop face processing thread
    try:
        if face_processing_thread:
            if face_processing_thread.is_alive():
                face_thread_kill = True
                face_processing_thread.join(timeout=5.0)

                if face_processing_thread.is_alive():
                    errors.append("Failed to stop face thread within timeout")
                else:
                    success_messages.append("Face processing stopped successfully")
                    face_thread_kill = False
                    face_processing_thread = None
            else:
                face_processing_thread = None
                success_messages.append("Face thread was already stopped")
        else:
            success_messages.append("Face thread not started")
    except Exception as e:
        errors.append(f"Error stopping face thread: {str(e)}")
        logger.error(f"Error stopping face processing thread: {str(e)}")

    # Handle response
    if errors:
        return make_response(
            {"errors": errors, "successes": success_messages},
            500 if len(errors) == 2 else 207,
        )  # 207 Multi-Status if partial success

    if not success_messages:
        return make_response("No processing threads were running", 200)

    return make_response(
        {"message": "All streams stopped successfully", "details": success_messages},
        200,
    )


# TODO: add a route to view both streams at the same time


@stream_viewer.route("/face_stream_start")
def face_stream_start():
    global face_processing_thread

    if face_processing_thread and face_processing_thread.is_alive():
        return make_response("Face processing thread already running", 409)

    sessionId = uuid.uuid4()
    face_processing_thread = threading.Thread(
        target=process_stream_face,
        args=(FACE_STREAM_URL, sessionId),
    )
    face_processing_thread.start()
    return make_response(f"Face processing started for session {sessionId}", 200)


@stream_viewer.route("/face_stream_stop")
def face_stream_stop():
    global face_processing_thread, face_thread_kill

    try:
        if not face_processing_thread:
            return make_response("No processing thread running", 200)

        if not face_processing_thread.is_alive():
            face_processing_thread = None
            return make_response("Thread already stopped", 200)

        face_thread_kill = True
        face_processing_thread.join(timeout=5.0)

        if face_processing_thread.is_alive():
            return make_response("Failed to stop thread within timeout", 500)

        face_thread_kill = False
        face_processing_thread = None
        return make_response("Face processing stopped successfully", 200)

    except Exception as e:
        logger.error(f"Error stopping face processing thread: {str(e)}")
        return make_response(f"Error stopping thread: {str(e)}", 500)


@stream_viewer.route("/face_stream_view")
def face_stream_view():
    global face_stream_data_buffer

    def generate():
        while True:
            try:
                while not face_stream_data_buffer.empty():
                    frame = face_stream_data_buffer.get_nowait()
                    yield f"data: {json.dumps(frame)}\n\n"
            except queue.Empty:
                pass
            time.sleep(0.1)  # Prevent CPU thrashing

    return Response(generate(), mimetype="text/event-stream")


@stream_viewer.route("/body_stream_start")
def body_stream_start():
    global body_processing_thread

    if body_processing_thread and body_processing_thread.is_alive():
        return make_response("Body processing thread already running", 409)

    sessionId = uuid.uuid4()
    body_processing_thread = threading.Thread(
        target=process_stream_body, args=(BODY_STREAM_URL, sessionId)
    )
    body_processing_thread.start()
    return make_response(f"Body processing started for session {sessionId}", 200)


@stream_viewer.route("/body_stream_stop")
def body_stream_stop():
    global body_processing_thread, body_thread_kill

    try:
        if not body_processing_thread:
            return make_response("No processing thread running", 200)

        if not body_processing_thread.is_alive():
            body_processing_thread = None
            return make_response("Thread already stopped", 200)

        body_thread_kill = True
        body_processing_thread.join(timeout=5.0)  # Add timeout to prevent hanging

        if body_processing_thread.is_alive():
            return make_response("Failed to stop thread within timeout", 500)

        body_thread_kill = False
        body_processing_thread = None
        return make_response("Processing stopped successfully", 200)

    except Exception as e:
        logger.error(f"Error stopping body processing thread: {str(e)}")
        return make_response(f"Error stopping thread: {str(e)}", 500)


@stream_viewer.route("/body_stream_view")
def body_stream_view():
    global body_stream_data_buffer

    def generate():
        while True:
            try:
                while not body_stream_data_buffer.empty():
                    frame = body_stream_data_buffer.get_nowait()
                    yield f"data: {json.dumps(frame)}\n\n"
            except queue.Empty:
                pass
            time.sleep(0.1)  # Prevent CPU thrashing

    return Response(generate(), mimetype="text/event-stream")


# TODO: Save to the database if we are on a different second based on server time
# Store the last second we saved with somewhere globally to know if it changed
@stream_viewer.route("/obd_data", methods=["POST"])
def receive_obd_data():
    global obd_data_buffer, current_session_id
    try:
        data = request.get_json()
        required_fields = ["speed", "rpm", "timestamp"]

        if not current_session_id:
            return make_response("No active session found", 400)
        
        if all(field in data for field in required_fields):
            # Process data for live view
            view_data = {
                "timestamp": data["timestamp"],
                "speed": data.get("speed", -1),
                "rpm": data.get("rpm", -1)
            }
            
            if "check_engine_on" in data:
                view_data["check_engine_on"] = data["check_engine_on"]
                
            if "num_dtc_codes" in data:
                view_data["num_dtc_codes"] = data["num_dtc_codes"]
                
            if "dtc_codes" in data:
                view_data["dtc_codes"] = data["dtc_codes"]
            
            # Update live view buffer
            if obd_data_buffer.full():
                obd_data_buffer.get_nowait()
            obd_data_buffer.put(view_data)
            
            process_obd_data(data, current_session_id)
            
            return make_response("Data received and processed", 200)
        else:
            if "timestamp" not in data:
                return make_response("Missing 'timestamp' field", 400)
            
            if "speed" not in data:
                return make_response("Missing 'speed' field", 400)
            
            if "rpm" not in data:
                return make_response("Missing 'rpm' field", 400)
            
        return make_response("Missing required fields", 400)
    except Exception as e:
        logger.error(f"Error processing OBD data: {str(e)}")
        return make_response("Error processing data", 400)


@stream_viewer.route("/obd_stream_view")
def obd_stream_view():
    global obd_data_buffer

    def generate():
        while True:
            try:
                while not obd_data_buffer.empty():
                    data = obd_data_buffer.get_nowait()
                    yield f"data: {json.dumps(data)}\n\n"
            except queue.Empty:
                pass
            time.sleep(0.1)

    return Response(generate(), mimetype="text/event-stream")
