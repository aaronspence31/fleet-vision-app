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
from firestore import body_drive_sessions, face_drive_sessions
from firebase_admin import firestore
from flask import Blueprint, Response, make_response
import dlib
from imutils import face_utils

realtime_camera_stream_handling = Blueprint("realtime_camera_stream_handling", __name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
# FACE_STREAM_URL = "http://172.20.10.3/stream"  # ai thinker hotspot aaron
FACE_STREAM_URL = "http://192.168.0.105/stream"  # ai thinker home wifi aaron
# BODY_STREAM_URL = "http://172.20.10.8/stream"  # ai thinker hotspot aaron
BODY_STREAM_URL = "http://192.168.0.104/stream"  # wrover home wifi aaron
CLIP_MODEL_NAME = "ViT-L/14"
CLIP_INPUT_SIZE = 224
# Using linear regression model trained here for prediction based on CLIP feature output:
# https://github.com/zahid-isu/DriveCLIP/tree/main?tab=readme-ov-file
CLIP_MODEL_PATH_BODY = "dmd29_vitbl14-hypc_429_1000_ft.pkl"
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.28
NUM_SECONDS_BEFORE_STORE_IN_DB = 3

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

# Setup model directory
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "models")

# Setup dlib face landmark predictor to find eyes and mouth
try:
    landmark_predictor = dlib.shape_predictor(
        os.path.join(model_dir, "shape_predictor_68_face_landmarks.dat")
    )
    logger.info("Dlib Face Landmark model loaded successfully")
except Exception as e:
    logger.error(f"Error loading Dlib Face Landmark model: {str(e)}")
    raise

# Setup CLIP global variables
clip_model = None
clip_preprocess = None
clip_classifier_body = None
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"CLIP Using device: {device}")

# Initialize CLIP components
try:
    clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device)
    clip_model.eval()
    clip_classifier_body = joblib.load(os.path.join(model_dir, CLIP_MODEL_PATH_BODY))
    logger.info("CLIP model and classifier loaded successfully")
except Exception as e:
    logger.error(f"Error loading CLIP model: {str(e)}")
    raise

# Map indices to labels for body stream regression model output
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

# Stream processing buffers, thread management variables and counters
# Setup buffers to hold the predictions data for all frames in the current second
face_stream_cursecond_buffer = []
body_stream_cursecond_buffer = []
# Thread safe buffers for viewing the video streams and predictions in real-time
face_stream_data_buffer = Queue(maxsize=1)
body_stream_data_buffer = Queue(maxsize=1)
# Setup db streaming buffers
face_frame_buffer_db = []
body_frame_buffer_db = []
# Setup threads and thread kill flags
face_processing_thread = None
face_thread_kill = False
body_processing_thread = None
body_thread_kill = False
# Store the current frame count for each stream
frame_count_face = 0
frame_count_body = 0


# Face stream helper function to save to the DB
# Each second classification will be stored in the format below
# face_drive_sessions/<sessionId>/face_drive_session_classifications/<timestamp>
# there will be a eye_classification and mouth_classification field with the final classification for that second
# Eyes State classifications are Eyes Open, Eyes Closed or Unknown
# Mouth State classifications are Mouth Open, Mouth Closed or Unknown
def save_face_frames_to_firestore(sessionId):
    """
    Saves classification data to Firestore under a document named after sessionId
    in the 'face_drive_sessions' collection. If the session doc does not exist, it is created.
    Otherwise, we update the existing doc with new timestamped classifications.
    """
    global face_frame_buffer_db

    db_start_time = time.time()
    logger.debug("FACE_STREAM: Saving face frames to Firestore")

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
        logger.debug(f"FACE_STREAM: Created new session doc for session ID {sessionId}")

    # 2) Write each (timestamp, classification) as a doc in 'face_drive_session_classifications'
    for record in frame_data:
        ts = record["timestamp"]  # integer second or unique timestamp
        eye_label = record["eye_classification"]  # eye classification
        mouth_label = record["mouth_classification"]  # mouth classification

        # Document ID = timestamp; store both timestamp and classifications
        doc_ref.collection("face_drive_session_classifications").document(str(ts)).set(
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


# Main face stream processing function
# Eyes State Predictions are Eyes Open, Eyes Closed or Unknown
# Mouth State Predictions are Mouth Open, Mouth Closed or Unknown
# Used paper here https://pmc.ncbi.nlm.nih.gov/articles/PMC11398282/
def process_stream_face(url, sessionId):
    """
    Main face stream processing function.

    This function reads video frames from the given `url` (e.g., an ESP32 camera),
    detects faces and facial landmarks (eyes/mouth), then calculates:
      1) Eye Aspect Ratio (EAR) to classify the eyes as 'Open' or 'Closed'
      2) Mouth Aspect Ratio (MAR) to classify the mouth as 'Open' or 'Closed'

    The function logs the timing of each step (frame capture, face detection, landmark
    detection, EAR/MAR calculation, encoding) for every frame. It also buffers these
    classifications each second and performs a majority vote to yield a final
    (Eyes, Mouth) classification for that second. That final classification is stored
    for Firestore insertion and also pushed to the front-end once per second.

    Args:
        url (str): The camera streaming endpoint.
        sessionId (str): A unique session identifier for storing results in Firestore.
    """
    global frame_count_face, face_stream_data_buffer, face_thread_kill
    global face_frame_buffer_db, face_stream_cursecond_buffer

    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    from collections import Counter

    # Track the integer second for majority-voting logic
    last_second = None

    while not face_thread_kill:
        # Record the start time for timing the entire pipeline
        pipeline_start_time = time.time()

        # Initialize timing placeholders for each step
        frame_capture_time = 0
        face_detect_time = 0
        dlib_time = 0
        ear_time = 0
        mar_time = 0
        encoding_time = 0

        # Placeholder for DB writes, which only occur once per second
        db_time_placeholder = 0

        # Determine the current integer second
        now_second = int(time.time())

        # ------------------------------------------------------------------
        # 1) If we've rolled over to a new second, finalize the old second
        # ------------------------------------------------------------------
        if last_second is not None and now_second != last_second:
            if len(face_stream_cursecond_buffer) > 0:
                # Gather the per-frame classifications from the previous second
                eye_labels = [item["eye"] for item in face_stream_cursecond_buffer]
                mouth_labels = [item["mouth"] for item in face_stream_cursecond_buffer]

                # Majority vote for eyes
                eye_counts = Counter(eye_labels)
                majority_eye, _ = eye_counts.most_common(1)[0]

                # Majority vote for mouth
                mouth_counts = Counter(mouth_labels)
                majority_mouth, _ = mouth_counts.most_common(1)[0]

                # Time how long DB operations take (if we do them)
                db_start_time = time.time()

                # Build a record for Firestore
                record = {
                    "timestamp": last_second,
                    "eye_classification": majority_eye,
                    "mouth_classification": majority_mouth,
                }

                # Store to the DB buffer
                if len(face_frame_buffer_db) < NUM_SECONDS_BEFORE_STORE_IN_DB:
                    face_frame_buffer_db.append(record)
                if len(face_frame_buffer_db) >= NUM_SECONDS_BEFORE_STORE_IN_DB:
                    save_face_frames_to_firestore(sessionId)

                # Measure the DB operation time
                db_elapsed = (time.time() - db_start_time) * 1000

                # Optionally push a final entry to the front-end with the aggregated classification
                current_entry = None
                if face_stream_data_buffer.full():
                    current_entry = face_stream_data_buffer.get_nowait()

                final_image = current_entry["image"] if current_entry else None
                face_stream_data_buffer.put(
                    {
                        "image": final_image,
                        "eye_prediction": majority_eye,
                        "mouth_prediction": majority_mouth,
                        "ear_score": None,  # optional
                        "mar_score": None,  # optional
                        "frame_number": None,
                        "timestamp": last_second,
                        "processing_time": None,
                    }
                )

                # Clear out the old second’s buffer
                face_stream_cursecond_buffer.clear()

                # Log that we've finalized this second
                logger.debug(
                    f"FACE_STREAM: Finalizing second {last_second}, "
                    f"Eyes='{majority_eye}', Mouth='{majority_mouth}', "
                    f"DB write time={db_elapsed:.2f}ms"
                )

        # Initialize last_second if needed, or update it if the second changed
        if last_second is None:
            last_second = now_second
        elif now_second != last_second:
            last_second = now_second

        # ------------------------------------------------------------------
        # 2) Per-frame processing: capture, detection, landmark, etc.
        # ------------------------------------------------------------------
        capture_start = time.time()
        success, frame = cap.read()
        frame_capture_time = (time.time() - capture_start) * 1000

        if not success:
            logger.error(f"FACE_STREAM: Failed to read frame from {url}")
            time.sleep(0.1)
            continue

        # Increment the frame count
        frame_count_face += 1

        # Default classifications/metrics
        eye_classification = "Unknown"
        mouth_classification = "Unknown"
        ear_score = None
        mar_score = None

        # Convert to grayscale and detect faces
        face_detect_start = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 1)
        face_detect_time = (time.time() - face_detect_start) * 1000

        # Copy for drawing/visualization
        frame_with_boxes = frame.copy()

        if len(faces) == 0:
            # If no face found, we label "Unknown"
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
            # Identify largest face
            largest_face = max(faces, key=lambda fc: fc[2] * fc[3])
            fx, fy, fw, fh = largest_face
            cv2.rectangle(
                frame_with_boxes, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2
            )

            # Landmark detection for eyes/mouth
            dlib_start = time.time()
            dlib_rect = dlib.rectangle(int(fx), int(fy), int(fx + fw), int(fy + fh))
            shape = landmark_predictor(gray, dlib_rect)
            shape_np = face_utils.shape_to_np(shape)
            dlib_time = (time.time() - dlib_start) * 1000

            # (a) Eye Aspect Ratio
            ear_start = time.time()
            left_eye_pts = shape_np[36:42]
            right_eye_pts = shape_np[42:48]

            # Draw small dots on each eye landmark
            for x, y in np.concatenate((left_eye_pts, right_eye_pts)):
                cv2.circle(frame_with_boxes, (x, y), 1, (255, 255, 0), -1)

            left_ear = compute_EAR(left_eye_pts)
            right_ear = compute_EAR(right_eye_pts)
            avg_ear = (left_ear + right_ear) / 2.0
            ear_score = avg_ear

            # Classify eyes
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
            eye_text = f"{eye_classification} ({avg_ear:.2f})"
            cv2.putText(
                frame_with_boxes,
                eye_text,
                (ex, ey - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                eye_color,
                2,
            )

            # (b) Mouth Aspect Ratio
            mar_start = time.time()
            mouth_pts = shape_np[60:68]
            mar_score = compute_MAR(mouth_pts)

            # Classify mouth
            mouth_classification = (
                "Mouth Open" if mar_score > MAR_THRESHOLD else "Mouth Closed"
            )
            mar_time = (time.time() - mar_start) * 1000

            # Draw small dots on each mouth landmark
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

        # Encode frame for streaming
        encode_start = time.time()
        _, buffer = cv2.imencode(".jpg", frame_with_boxes)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")
        encoding_time = (time.time() - encode_start) * 1000

        # 8) Accumulate these classifications to do majority voting at the second boundary
        face_stream_cursecond_buffer.append(
            {
                "eye": eye_classification,
                "mouth": mouth_classification,
            }
        )

        # 9) Send partial data (image + placeholder classification) to the front-end
        if face_stream_data_buffer.full():
            face_stream_data_buffer.get_nowait()

        total_time = (time.time() - pipeline_start_time) * 1000
        face_stream_data_buffer.put(
            {
                "image": frame_base64,
                "eye_prediction": None,  # final classification done only once per second
                "mouth_prediction": None,  # final classification
                "ear_score": ear_score,  # optional
                "mar_score": mar_score,  # optional
                "frame_number": frame_count_face,
                "timestamp": now_second,
                "processing_time": total_time,
            }
        )

        # ----------------------------------------------------------------
        # 10) Per-frame logging
        # ----------------------------------------------------------------
        # We keep a DB time placeholder since DB writes only happen once per second.
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
        # Use a placeholder for DB time, since we only write to Firestore at the second boundary
        timing_log += (
            f"  - Frame Encoding: {encoding_time:.2f}ms\n"
            f"  - DB Time (placeholder): {db_time_placeholder:.2f}ms\n"
            f"  - Total Pipeline: {total_time:.2f}ms\n"
        )
        logger.debug(timing_log)


# Body stream helper function for saving to DB
# Each second classification will be stored in the format below
# body_drive_sessions/<sessionId>/body_drive_session_classifications/<timestamp>
# there will be a single classification field with the final classification for that second
# Expect body_stream_index_to_label classifications and Unknown as possible classifications
def save_body_frames_to_firestore(sessionId):
    """
    Saves classification data to Firestore under a document named after sessionId
    in the 'body_drive_sessions' collection. If the session doc does not exist, it is created.
    Otherwise, we update the existing doc with new timestamped classifications.
    """
    global body_frame_buffer_db

    db_start_time = time.time()
    logger.debug("BODY_STREAM: Saving body frames to Firestore")

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
        logger.debug(f"BODY_STREAM: Created new session doc for session ID {sessionId}")

    # 2) Write each (timestamp, classification) as a doc in 'body_drive_session_classifications'
    for record in frame_data:
        ts = record["timestamp"]  # integer second or unique timestamp
        label = record["classification"]  # your final classification

        # Document ID = timestamp; store both timestamp and classification
        doc_ref.collection("body_drive_session_classifications").document(str(ts)).set(
            {"timestamp": ts, "classification": label}
        )

    db_time = (time.time() - db_start_time) * 1000  # milliseconds
    logger.info(f"BODY_STREAM: Database save completed in {db_time:.2f}ms")

    # 3) Clear the local buffer and reset
    body_frame_buffer_db = []


# Body stream helper function for preprocess inputs to CLIP
def preprocess_frame_clip(frame, preprocess):
    # Convert BGR to RGB and to PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    # Apply CLIP preprocessing
    processed = preprocess(pil_image)
    # Add batch dimension
    return processed.unsqueeze(0)


# Body stream helper function to detect person in frame - NOT USED RIGHT NOW
# Detect a person in frame so we predict Unknown if there is no person detected rather
# than confidently predicting safe driving when there is noone in the frame
def detect_person_in_frame(frame, scale_factor=1.2, min_neighbors=1):
    start_time = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Body stream order: profile face -> upper body -> frontal face
    profile_faces = profileFaceCascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors
    )
    if len(profile_faces) > 0:
        detection_time = (time.time() - start_time) * 1000
        logger.debug(
            f"BODY_STREAM: Person detection completed in {detection_time:.2f}ms (profile face detected)"
        )
        return True

    upper_bodies = upperBodyCascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors
    )
    if len(upper_bodies) > 0:
        detection_time = (time.time() - start_time) * 1000
        logger.debug(
            f"BODY_STREAM: Person detection completed in {detection_time:.2f}ms (upper body detected)"
        )
        return True

    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors
    )
    detection_time = (time.time() - start_time) * 1000
    logger.debug(
        f"BODY_STREAM: Person detection completed in {detection_time:.2f}ms (Person {'detected' if len(faces) > 0 else 'not detected'})"
    )
    return len(faces) > 0


# Main body stream processing function
# Expect body_stream_index_to_label classifications and Unknown as possible predictions on each frame
def process_stream_body(url, sessionId):
    """
    Main body stream processing function.

    This function reads video frames from the given `url` (for example, an ESP32 camera),
    applies a CLIP-based model plus a linear regression classifier to predict driver behavior,
    and then:
      1) Logs timing for each pipeline step (capture, preprocessing, feature extraction, prediction, encoding).
      2) Accumulates predictions per second in body_stream_cursecond_buffer.
      3) Once the second ends, performs a majority vote to produce a "final" classification for that second.
      4) Stores that final classification in the Firestore buffer and optionally pushes it to the front-end.

    Args:
        url (str): The camera streaming endpoint.
        sessionId (str): Unique identifier for the driving session (used for Firestore).
    """

    global frame_count_body, clip_model, clip_preprocess, clip_classifier_body
    global body_stream_data_buffer, body_frame_buffer_db, body_thread_kill
    global body_stream_cursecond_buffer

    # Ensure CLIP model and components are initialized before proceeding
    if not all([clip_model, clip_preprocess, clip_classifier_body]):
        logger.error("BODY_STREAM: CLIP components not initialized")
        return

    # Open the video capture
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Attempt to reduce buffering

    # Track the integer second so we know when a new second starts.
    # We will accumulate per-frame predictions in body_stream_cursecond_buffer during that second.
    last_second = None

    while not body_thread_kill:
        # Record the start time for timing the entire pipeline
        pipeline_start_time = time.time()

        # Initialize timing placeholders for each step
        frame_capture_time = 0
        preprocess_time = 0
        feature_extraction_time = 0
        prediction_time = 0
        encoding_time = 0

        # We only write to the DB once per second (in the aggregator),
        # so this is just a placeholder for the per-frame log:
        db_time_placeholder = 0

        # Determine the current integer second (used for majority-voting logic)
        now_second = int(time.time())

        # ---------------------------------------------------------
        # 1) Finalize predictions if we've rolled over to a new second
        # ---------------------------------------------------------
        if last_second is not None and now_second != last_second:
            # If we have predictions in the buffer for the previous second,
            # compute a majority vote and store/publish the final classification.
            if len(body_stream_cursecond_buffer) > 0:
                counts = Counter(body_stream_cursecond_buffer)
                majority_label, _ = counts.most_common(1)[0]

                # Record the time to measure DB write overhead (if it occurs)
                db_start_time = time.time()

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

                # Measure time spent in DB logic (including function call)
                db_elapsed = (time.time() - db_start_time) * 1000

                # 3) Optionally also push a “final” entry to the front-end
                current_entry = None
                if body_stream_data_buffer.full():
                    current_entry = body_stream_data_buffer.get_nowait()

                # If the stream viewer code already got the entry, image can be None
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

                # Clear the buffer for the old second
                body_stream_cursecond_buffer.clear()

                # Log that we finalized this second
                logger.debug(
                    f"BODY_STREAM: Finalizing second {last_second}, "
                    f"majority label='{majority_label}', DB write time={db_elapsed:.2f}ms"
                )

        # Initialize last_second if needed, or update it if the second rolled over
        if last_second is None:
            last_second = now_second
        elif now_second != last_second:
            last_second = now_second

        # ---------------------------------------------------------
        # 2) Per-frame pipeline begins (capture, predict, etc.)
        # ---------------------------------------------------------

        # (a) Frame capture
        capture_start = time.time()
        success, frame = cap.read()
        frame_capture_time = (time.time() - capture_start) * 1000

        if not success:
            logger.error(f"BODY_STREAM: Failed to read frame from {url}")
            time.sleep(1.0)
            continue

        # Increment the total frame counter
        frame_count_body += 1

        # Default label and probability
        prediction_label = "Unknown"
        prob_score = None

        # Create a copy for visualization
        frame_with_text = frame.copy()

        # Currently we are not actually detecting a person; forcibly set True
        # If we did want real detection, we'd use a cascade or YOLO to confirm a person is present.
        person_detected = True

        if person_detected:
            try:
                # (b) Preprocessing for CLIP
                preprocess_start = time.time()
                processed = preprocess_frame_clip(frame, clip_preprocess).to(device)
                preprocess_time = (time.time() - preprocess_start) * 1000

                # (c) Feature extraction with CLIP
                feature_start = time.time()
                with torch.no_grad():
                    features = clip_model.encode_image(processed)
                features = features.cpu().numpy()
                feature_extraction_time = (time.time() - feature_start) * 1000

                # (d) Model prediction
                # Using linear regression model trained here for prediction based on CLIP feature output:
                # https://github.com/zahid-isu/DriveCLIP/tree/main?tab=readme-ov-file
                prediction_start = time.time()
                prediction = int(clip_classifier_body.predict(features)[0])
                prob_score = clip_classifier_body.predict_proba(features)[0][prediction]
                prediction_label = body_stream_index_to_label.get(prediction, "Unknown")
                prediction_time = (time.time() - prediction_start) * 1000

                # Set text color based on whether it's "Driving Safely," "Unknown," or other unsafe behavior
                if prediction_label == "Driving Safely":
                    text_color = (0, 255, 0)  # Green
                elif prediction_label == "Unknown":
                    text_color = (0, 165, 255)  # Orange
                else:
                    text_color = (0, 0, 255)  # Red

                # Draw the prediction + probability in the top-right corner of the frame
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

                # (e) Frame encoding for streaming / front-end
                encode_start = time.time()
                _, buffer = cv2.imencode(".jpg", frame_with_text)
                frame_base64 = base64.b64encode(buffer).decode("utf-8")
                encoding_time = (time.time() - encode_start) * 1000

                # Calculate total pipeline time
                total_time = (time.time() - pipeline_start_time) * 1000

                # Log all timing steps in a concise, structured manner.
                # Note that "db_time_placeholder" is always ~0 except on finalizing a second.
                logger.debug(
                    f"BODY_STREAM: Frame {frame_count_body} pipeline:\n"
                    f"  - Capture: {frame_capture_time:.2f}ms\n"
                    f"  - Preprocessing: {preprocess_time:.2f}ms\n"
                    f"  - Feature Extraction: {feature_extraction_time:.2f}ms\n"
                    f"  - Prediction: {prediction_time:.2f}ms\n"
                    f"  - Encoding: {encoding_time:.2f}ms\n"
                    f"  - DB Time (placeholder): {db_time_placeholder:.2f}ms\n"
                    f"  - Total: {total_time:.2f}ms\n"
                    f"  - Classification: {prediction_label}\n"
                    f"  - Probability: {prob_score:.3f}"
                )

            except Exception as e:
                # If an error occurs, default label remains "Unknown"
                logger.error(f"BODY_STREAM: Error in inference: {str(e)}")

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
                    (0, 165, 255),
                    thickness,
                )

                _, buffer = cv2.imencode(".jpg", frame_with_text)
                frame_base64 = base64.b64encode(buffer).decode("utf-8")

        else:
            # If no person is detected (not currently used in practice)
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
                (0, 165, 255),
                thickness,
            )

            encode_start = time.time()
            _, buffer = cv2.imencode(".jpg", frame_with_text)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")
            encoding_time = (time.time() - encode_start) * 1000

            total_time = (time.time() - pipeline_start_time) * 1000
            logger.debug(
                f"BODY_STREAM: Frame {frame_count_body} pipeline (no person detected):\n"
                f"  - Capture: {frame_capture_time:.2f}ms\n"
                f"  - Encoding: {encoding_time:.2f}ms\n"
                f"  - Total: {total_time:.2f}ms\n"
                f"  - Prediction: Unknown (No person detected)"
            )

        # ----------------------------------------------------------------
        # 3) Accumulate the label for majority vote later
        # ----------------------------------------------------------------
        body_stream_cursecond_buffer.append(prediction_label)

        # ----------------------------------------------------------------
        # 4) Send partial results (image + placeholder prediction) to front-end
        # ----------------------------------------------------------------
        # The final classification is assigned once the second ends.
        if body_stream_data_buffer.full():
            body_stream_data_buffer.get_nowait()

        total_time = (time.time() - pipeline_start_time) * 1000
        body_stream_data_buffer.put(
            {
                "image": frame_base64,
                "prediction": None,  # Hide final classification until we do majority vote
                "probability": None,  # Hide probability until final
                "frame_number": frame_count_body,
                "timestamp": now_second,
                "processing_time": total_time,
            }
        )


# PUT ALL ROUTES BELOW -----------------------------------------------------------------------


# Start both stream processing functions as part of the same session
@realtime_camera_stream_handling.route("/both_streams_start")
def both_streams_start():
    global body_processing_thread, face_processing_thread

    if body_processing_thread and body_processing_thread.is_alive():
        return make_response("Body processing thread already running", 409)

    if face_processing_thread and face_processing_thread.is_alive():
        return make_response("Face processing thread already running", 409)

    sessionId = uuid.uuid4()
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


# Attempt to stop both processing functions
@realtime_camera_stream_handling.route("/both_streams_stop")
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


@realtime_camera_stream_handling.route("/face_stream_view")
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
            time.sleep(0.05)

    return Response(generate(), mimetype="text/event-stream")


@realtime_camera_stream_handling.route("/body_stream_view")
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
            time.sleep(0.05)

    return Response(generate(), mimetype="text/event-stream")


@realtime_camera_stream_handling.route("/face_stream_start")
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


@realtime_camera_stream_handling.route("/face_stream_stop")
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


@realtime_camera_stream_handling.route("/body_stream_start")
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


@realtime_camera_stream_handling.route("/body_stream_stop")
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
