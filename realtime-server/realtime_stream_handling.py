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
from imutils import face_utils
from zeroconf import Zeroconf
import socket

# This file defines the Flask blueprints for handling the real-time camera and OBD streams.
realtime_camera_stream_handling = Blueprint("realtime_camera_stream_handling", __name__)
realtime_obd_stream_handling = Blueprint("realtime_obd_stream_handling", __name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# We must always set the current session ID and name if there is one whenever we start a new face or body processing thread
# This is because the OBD code will need to use the current session ID to write to the DB and it might want the name too
current_session_id = None
current_session_name = None
# This constant applies to both camera stream and OBD DB saves
NUM_SECONDS_BEFORE_STORE_IN_DB = 1

######################################################################################
######################################################################################
##### CAMERA STREAM HANDLING CODE BELOW ##############################################
######################################################################################
######################################################################################

# Constants
# These work when on wifi, may break when on hotspot
# Ideally it will still work as long as you have maximize compatibility enabled on the hotspot
# It works on a hotspot with maximize compatibility enabled for me
FACE_STREAM_MDNS_NAME = "esp32face.local"
BODY_STREAM_MDNS_NAME = "esp32body.local"
CLIP_MODEL_NAME = "ViT-L/14"
CLIP_INPUT_SIZE = 224
# Using linear regression model trained here for prediction based on CLIP feature output:
# https://github.com/zahid-isu/DriveCLIP/tree/main?tab=readme-ov-file
CLIP_MODEL_PATH_BODY = "dmd29_vitbl14-hypc_429_1000_ft.pkl"
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.28

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
face_stream_cursecond_buffer = Queue()
body_stream_cursecond_buffer = Queue()
# Thread safe buffers for viewing the video streams and predictions in real-time
face_stream_data_buffer = Queue(maxsize=1)
body_stream_data_buffer = Queue(maxsize=1)
# Setup db streaming buffers
face_frame_buffer_db = Queue()
body_frame_buffer_db = Queue()
# Setup threads and thread kill flags
face_processing_thread = None
face_thread_kill = False
body_processing_thread = None
body_thread_kill = False
# Store the current frame count for each stream
frame_count_face = 0
frame_count_body = 0


# Helper function used by stream starters to resolve the mDNS name to an IP address
def resolve_mdns_via_zeroconf(hostname: str) -> str:
    """
    Attempt to resolve the given mDNS 'hostname' (like 'esp32face.local')
    to an IPv4 address using Zeroconf. Returns IP string or None on failure.
    """
    zc = Zeroconf()
    try:
        time.sleep(0.1)  # optionally let Zeroconf start up
        ip = socket.gethostbyname(hostname)
        logger.info(f"Resolved {hostname} => {ip}")
        return ip
    except Exception as e:
        logger.error(f"Failed to resolve {hostname} via Zeroconf: {e}")
        return None
    finally:
        zc.close()


# Face stream helper function to save to the DB
# Each face stream second classification will be stored in the format below
# Each session will also be given a created_at_value and optionally a session_name if one was provided when the session was made
# face_drive_sessions/<sessionId>/face_drive_session_classifications/<timestamp>
# there will be a eye_classification and mouth_classification field with the final classification for that second
# String Eyes State classifications are "Eyes Open", "Eyes Closed" or "Unknown"
# String Mouth State classifications are "Mouth Open", "Mouth Closed" or "Unknown"
def save_face_frames_to_firestore():
    """
    Saves classification data to Firestore under a document named after sessionId
    in the 'face_drive_sessions' collection. If the session doc does not exist, it is created.
    Otherwise, we update the existing doc with new timestamped classifications.
    """
    global face_frame_buffer_db, current_session_id, current_session_name

    # Record the overall function start
    overall_start_time = time.time()

    logger.debug("FACE_STREAM: Beginning save_face_frames_to_firestore")

    # Step 1) Copy the local buffer into a list
    copy_start_time = time.time()
    frame_data = []
    while not face_frame_buffer_db.empty():
        try:
            frame_data.append(face_frame_buffer_db.get_nowait())
        except queue.Empty:
            break
    copy_time = (time.time() - copy_start_time) * 1000

    # Step 2) Create or retrieve the Firestore doc for this session ID
    doc_check_start_time = time.time()
    doc_ref = face_drive_sessions.document(str(current_session_id))
    doc_snapshot = doc_ref.get()
    doc_check_time = (time.time() - doc_check_start_time) * 1000

    doc_create_start_time = None
    doc_create_time = 0
    if not doc_snapshot.exists:
        doc_create_start_time = time.time()
        doc_ref.set(
            {
                "session_id": str(current_session_id),
                "created_at": firestore.SERVER_TIMESTAMP,
                "session_name": current_session_name,
            }
        )
        doc_create_time = (time.time() - doc_create_start_time) * 1000
        logger.debug(
            f"FACE_STREAM: Created new session doc for session ID {current_session_id}"
        )

    # Step 3) Write each (timestamp, classification) as a doc in 'face_drive_session_classifications'
    write_start_time = time.time()
    for record in frame_data:
        ts = record["timestamp"]  # integer second or unique timestamp
        eye_label = record["eye_classification"]
        mouth_label = record["mouth_classification"]

        # Document ID = timestamp; store both timestamp and classifications
        doc_ref.collection("face_drive_session_classifications").document(str(ts)).set(
            {
                "timestamp": ts,
                "eye_classification": eye_label,
                "mouth_classification": mouth_label,
            }
        )
    write_time = (time.time() - write_start_time) * 1000

    # Step 4) Log the total time
    overall_time = (time.time() - overall_start_time) * 1000
    logger.debug(
        "FACE_STREAM: Firestore save completed\n"
        f"  - Copied buffer: {copy_time:.2f}ms\n"
        f"  - Doc check: {doc_check_time:.2f}ms\n"
        f"  - Doc create (if needed): {doc_create_time:.2f}ms\n"
        f"  - Writing {len(frame_data)} records: {write_time:.2f}ms\n"
        f"  - Total function time: {overall_time:.2f}ms"
    )


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
def process_stream_face(url):
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
    """
    global frame_count_face, face_stream_data_buffer, face_thread_kill
    global face_frame_buffer_db, face_stream_cursecond_buffer
    global current_session_id

    logger.info(
        f"FACE_STREAM: Starting face stream processing with sessionId: {current_session_id} and ESP32 URL: {url}"
    )

    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Track the integer second for majority-voting logic
    last_second = None
    frame_count_face = 0

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
            if not face_stream_cursecond_buffer.empty():
                buffer_data = []
                while not face_stream_cursecond_buffer.empty():
                    try:
                        buffer_data.append(face_stream_cursecond_buffer.get_nowait())
                    except queue.Empty:
                        break

                eye_labels = [item["eye"] for item in buffer_data]
                mouth_labels = [item["mouth"] for item in buffer_data]

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
                if face_frame_buffer_db.qsize() < NUM_SECONDS_BEFORE_STORE_IN_DB:
                    face_frame_buffer_db.put(record)
                if face_frame_buffer_db.qsize() >= NUM_SECONDS_BEFORE_STORE_IN_DB:
                    save_face_frames_to_firestore()

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
        face_stream_cursecond_buffer.put(
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
# Each body stream second classification will be stored in the format below
# Each session will also be given a created_at_value and optionally a session_name if one was provided when the session was made
# body_drive_sessions/<sessionId>/body_drive_session_classifications/<timestamp>
# there will be a single classification field with the final String classification for that second
# Expect  "Driving Safely", "Drinking beverage", "Adjusting hair, glasses, or makeup", "Talking on phone",
# "Reaching beside or behind", "Talking to passenger", "Texting/using phone", "Yawning" and "Unknown" as possible classifications
def save_body_frames_to_firestore():
    """
    Saves classification data to Firestore under a document named after sessionId
    in the 'body_drive_sessions' collection. If the session doc does not exist, it is created.
    Otherwise, we update the existing doc with new timestamped classifications.
    """
    global body_frame_buffer_db, current_session_id, current_session_name

    # Record the overall function start time
    overall_start_time = time.time()

    logger.debug("BODY_STREAM: Beginning save_body_frames_to_firestore")

    # Step 1) Copy the local buffer into a list
    copy_start_time = time.time()
    frame_data = []
    while not body_frame_buffer_db.empty():
        try:
            frame_data.append(body_frame_buffer_db.get_nowait())
        except queue.Empty:
            break
    copy_time = (time.time() - copy_start_time) * 1000

    # Step 2) Create or retrieve the doc for this session ID
    doc_check_start_time = time.time()
    doc_ref = body_drive_sessions.document(str(current_session_id))
    doc_snapshot = doc_ref.get()
    doc_check_time = (time.time() - doc_check_start_time) * 1000

    # Only create the doc if it doesn't exist
    doc_create_start_time = None
    doc_create_time = 0
    if not doc_snapshot.exists:
        doc_create_start_time = time.time()
        doc_ref.set(
            {
                "session_id": str(current_session_id),
                "created_at": firestore.SERVER_TIMESTAMP,
                "session_name": current_session_name,
            }
        )
        doc_create_time = (time.time() - doc_create_start_time) * 1000
        logger.debug(
            f"BODY_STREAM: Created new session doc for session ID {current_session_id}"
        )

    # Step 3) Write each (timestamp, classification) as a doc in 'body_drive_session_classifications'
    write_start_time = time.time()
    for record in frame_data:
        ts = record["timestamp"]
        label = record["classification"]

        # Document ID = timestamp; store both timestamp and classification
        doc_ref.collection("body_drive_session_classifications").document(str(ts)).set(
            {"timestamp": ts, "classification": label}
        )
    write_time = (time.time() - write_start_time) * 1000

    # Calculate the overall function time
    overall_time = (time.time() - overall_start_time) * 1000

    # Detailed log showing each step's duration
    logger.debug(
        "BODY_STREAM: Firestore save completed\n"
        f"  - Copied buffer: {copy_time:.2f}ms\n"
        f"  - Doc check: {doc_check_time:.2f}ms\n"
        f"  - Doc create (if needed): {doc_create_time:.2f}ms\n"
        f"  - Writing {len(frame_data)} records: {write_time:.2f}ms\n"
        f"  - Total function time: {overall_time:.2f}ms"
    )


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
def process_stream_body(url):
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
    """

    global frame_count_body, clip_model, clip_preprocess, clip_classifier_body
    global body_stream_data_buffer, body_frame_buffer_db, body_thread_kill
    global body_stream_cursecond_buffer, current_session_id

    logger.info(
        f"BODY_STREAM: Starting body stream processing with sessionId: {current_session_id} and ESP32 URL: {url}"
    )

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
    frame_count_body = 0

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
            if not body_stream_cursecond_buffer.empty():
                buffer_data = []
                while not body_stream_cursecond_buffer.empty():
                    try:
                        buffer_data.append(body_stream_cursecond_buffer.get_nowait())
                    except queue.Empty:
                        break

                counts = Counter(buffer_data)
                majority_label, _ = counts.most_common(1)[0]

                # Record the time to measure DB write overhead (if it occurs)
                db_start_time = time.time()

                # 1) Build a dict containing both timestamp and classification
                record = {
                    "timestamp": last_second,  # the integer second
                    "classification": majority_label,
                }

                # 2) Save it in the local DB buffer
                if body_frame_buffer_db.qsize() < NUM_SECONDS_BEFORE_STORE_IN_DB:
                    body_frame_buffer_db.put(record)
                if body_frame_buffer_db.qsize() >= NUM_SECONDS_BEFORE_STORE_IN_DB:
                    save_body_frames_to_firestore()

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
        body_stream_cursecond_buffer.put(prediction_label)

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


# PUT ALL CAMERA STREAM ROUTES BELOW -----------------------------------------------------------------------


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

    response = Response(generate(), mimetype="text/event-stream")
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Cache-Control", "no-cache")
    response.headers.add("Connection", "keep-alive")
    return response


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

    response = Response(generate(), mimetype="text/event-stream")
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Cache-Control", "no-cache")
    response.headers.add("Connection", "keep-alive")
    return response


######################################################################################
######################################################################################
##### OBD STREAM HANDLING CODE BELOW #################################################
######################################################################################
######################################################################################

# Thread safe buffer for viewing the obd streams in real-time
obd_stream_data_buffer = Queue(maxsize=1)
# Setup buffer to hold the obd data for all frames in the current second
obd_cursecond_buffer = Queue()
# Setup db streaming buffer
obd_frame_buffer_db = Queue()
# Keep track of the last processed second for the obd
last_second_obd = None
# Keep track of the current frame in the obd session
frame_count_obd = 0


# Each second's OBD frame data will be stored in the format below
# Each session will also be given a created_at_value and optionally a session_name if one was provided when the session was made
# obd_drive_sessions/<sessionId>/obd_drive_session_classifications/<timestamp>
# Each obd_drive_session_classifications doc will contain speed, rpm, check_engine_on and num_dtc_codes
# int speed and rpm will be 0 or greater if available and -1 if not available
# bool check_engine_on will be True/False if available or null if not available
# int num_dtc_codes will be 0 or greater if available or -1 if not available
def save_obd_frames_to_firestore():
    """
    Saves classification data to Firestore under a document named after sessionId
    in the 'obd_drive_sessions' collection. If the session doc does not exist, it is created.
    Otherwise, we update the existing doc with new timestamped obd data.
    """
    global obd_frame_buffer_db, current_session_id, current_session_name

    # Record the overall function start time
    overall_start_time = time.time()

    logger.debug("OBD_STREAM: Beginning save_obd_frames_to_firestore")

    # Step 1) Copy the local buffer so we don't mutate it while writing
    copy_start_time = time.time()
    frame_data = []
    while not obd_frame_buffer_db.empty():
        try:
            frame_data.append(obd_frame_buffer_db.get_nowait())
        except queue.Empty:
            break
    copy_time = (time.time() - copy_start_time) * 1000

    # Step 2) Create or retrieve the doc for this session ID
    doc_check_start_time = time.time()
    doc_ref = obd_drive_sessions.document(str(current_session_id))
    doc_snapshot = doc_ref.get()
    doc_check_time = (time.time() - doc_check_start_time) * 1000

    # Only create the doc if it doesn't exist
    doc_create_start_time = None
    doc_create_time = 0
    if not doc_snapshot.exists:
        doc_create_start_time = time.time()
        doc_ref.set(
            {
                "session_id": str(current_session_id),
                "created_at": firestore.SERVER_TIMESTAMP,
                "session_name": current_session_name,
            }
        )
        doc_create_time = (time.time() - doc_create_start_time) * 1000
        logger.debug(
            f"OBD_STREAM: Created new session doc for session ID {current_session_id}"
        )

    # Step 3) Write each (timestamp, classification) as a doc in 'obd_drive_session_classifications'
    write_start_time = time.time()
    for record in frame_data:
        ts = record["timestamp"]

        doc_data = {
            "timestamp": ts,
            "speed": record.get("speed", -1),
            "rpm": record.get("rpm", -1),
            "check_engine_on": record.get("check_engine_on", None),
            "num_dtc_codes": record.get("num_dtc_codes", -1),
        }

        # Optionally add dtc_codes and warning_lights if available
        if "dtc_codes" in record:
            doc_data["dtc_codes"] = record["dtc_codes"]
        if "warning_lights" in record:
            doc_data["warning_lights"] = record["warning_lights"]

        # Document ID = timestamp;
        doc_ref.collection("obd_drive_session_classifications").document(str(ts)).set(
            doc_data
        )
    write_time = (time.time() - write_start_time) * 1000

    # Calculate the overall function time
    overall_time = (time.time() - overall_start_time) * 1000

    # Detailed log showing each step's duration
    logger.debug(
        "OBD_STREAM: Firestore save completed\n"
        f"  - Copied buffer: {copy_time:.2f}ms\n"
        f"  - Doc check: {doc_check_time:.2f}ms\n"
        f"  - Doc create (if needed): {doc_create_time:.2f}ms\n"
        f"  - Writing {len(frame_data)} records: {write_time:.2f}ms\n"
        f"  - Total function time: {overall_time:.2f}ms"
    )


def process_obd_data(obd_data):
    """
    Process a single OBD data frame.

    This function performs two major tasks:
      1. If a new second has begun, it aggregates all OBD frames received during
         the previous second. It computes average values for numerical metrics
         and a majority vote for boolean flags. It then pushes the aggregated record
         into the Firestore DB buffer (and calls the DB save function if needed),
         while logging detailed pipeline timing.
      2. It then processes the current frame by adding it to the current-second buffer
         and updating the live stream buffer.

    The function logs detailed timing for:
      - Aggregation of per-frame data from the previous second.
      - The database-buffering operation.
      - The total processing time for each frame.
    """
    global last_second_obd, frame_count_obd, obd_frame_buffer_db, obd_cursecond_buffer

    # Record the start time of the pipeline for this frame.
    pipeline_start = time.time()

    # Get current server timestamp (in seconds)
    current_timestamp = int(time.time())
    logger.debug(f"OBD_STREAM: Received new OBD data at timestamp {current_timestamp}")

    # If we've rolled into a new second, finalize the previous second's data.
    if (
        last_second_obd is not None
        and current_timestamp != last_second_obd
        and not obd_cursecond_buffer.empty()
    ):
        finalize_start = time.time()
        # --- Aggregation Step ---
        # Extract lists for each measurement from the buffer.
        aggregation_start = time.time()
        buffer_data = []
        while not obd_cursecond_buffer.empty():
            try:
                buffer_data.append(obd_cursecond_buffer.get_nowait())
            except queue.Empty:
                break

        speed_labels = [item["speed"] for item in buffer_data]
        rpm_labels = [item["rpm"] for item in buffer_data]
        check_engine_on_labels = [item["check_engine_on"] for item in buffer_data]
        num_dtc_codes_labels = [item["num_dtc_codes"] for item in buffer_data]

        # Calculate averages for speed, rpm, and num_dtc_codes (ignoring invalid -1 values)
        valid_speeds = [speed for speed in speed_labels if speed != -1]
        avg_speed = sum(valid_speeds) / len(valid_speeds) if valid_speeds else -1

        valid_rpms = [rpm for rpm in rpm_labels if rpm != -1]
        avg_rpm = sum(valid_rpms) / len(valid_rpms) if valid_rpms else -1

        valid_dtc_codes = [code for code in num_dtc_codes_labels if code != -1]
        avg_num_dtc_codes = (
            sum(valid_dtc_codes) / len(valid_dtc_codes) if valid_dtc_codes else -1
        )

        # Determine the majority value for check_engine_on (ignoring None values)
        check_engine_on_counts = Counter(
            [val for val in check_engine_on_labels if val is not None]
        )
        majority_check_engine_on = (
            check_engine_on_counts.most_common(1)[0][0]
            if check_engine_on_counts
            else None
        )
        aggregation_elapsed = (time.time() - aggregation_start) * 1000
        logger.debug(
            f"OBD_STREAM: Aggregated previous second's data in {aggregation_elapsed:.2f}ms"
        )

        # --- DB Buffering Step ---
        db_start_time = time.time()
        record = {
            "timestamp": last_second_obd,
            "speed": avg_speed,
            "rpm": avg_rpm,
            "check_engine_on": majority_check_engine_on,
            "num_dtc_codes": avg_num_dtc_codes,
        }

        # Append the aggregated record to the DB buffer, and trigger save if threshold reached.
        if obd_frame_buffer_db.qsize() < NUM_SECONDS_BEFORE_STORE_IN_DB:
            obd_frame_buffer_db.put(record)
        if obd_frame_buffer_db.qsize() >= NUM_SECONDS_BEFORE_STORE_IN_DB:
            save_obd_frames_to_firestore()
        db_elapsed = (time.time() - db_start_time) * 1000

        finalize_elapsed = (time.time() - finalize_start) * 1000

        logger.info(
            f"OBD_STREAM: Finalized second {last_second_obd} in {finalize_elapsed:.2f}ms "
            f"(DB op: {db_elapsed:.2f}ms). Aggregated values - Speed: {avg_speed}, RPM: {avg_rpm}, "
            f"Check Engine On: {majority_check_engine_on}, Num DTC Codes: {avg_num_dtc_codes}"
        )

    # Update the last processed second if necessary.
    if last_second_obd is None or current_timestamp != last_second_obd:
        last_second_obd = current_timestamp

    # Increment the frame counter (using frame_count_obd for OBD frames).
    frame_count_obd += 1

    # --- Per-Frame Processing ---
    # Add current frame data to the per-second buffer.
    obd_cursecond_buffer_entry = {
        "speed": obd_data.get("speed", -1),
        "rpm": obd_data.get("rpm", -1),
        "check_engine_on": obd_data.get("check_engine_on", None),
        "num_dtc_codes": obd_data.get("num_dtc_codes", -1),
    }
    obd_cursecond_buffer.put(obd_cursecond_buffer_entry)

    # Build a live-stream buffer entry including a frame counter.
    obd_stream_buffer_entry = {
        "speed": obd_data.get("speed", -1),
        "rpm": obd_data.get("rpm", -1),
        "check_engine_on": obd_data.get("check_engine_on", None),
        "num_dtc_codes": obd_data.get("num_dtc_codes", -1),
        "timestamp": current_timestamp,
        "frame_number": frame_count_obd,
    }

    # Ensure the live-stream buffer has space; if full, discard the oldest entry.
    if obd_stream_data_buffer.full():
        obd_stream_data_buffer.get_nowait()
    obd_stream_data_buffer.put(obd_stream_buffer_entry)

    total_pipeline_time = (time.time() - pipeline_start) * 1000
    logger.info(
        f"OBD_STREAM: Processed frame {frame_count_obd} in {total_pipeline_time:.2f}ms - "
        f"Speed: {obd_data.get('speed', -1)}, RPM: {obd_data.get('rpm', -1)}, "
        f"Check Engine On: {obd_data.get('check_engine_on', None)}, Num DTC Codes: {obd_data.get('num_dtc_codes', -1)}"
    )


# PUT ALL OBD STREAM ROUTES BELOW -----------------------------------------------------------------------


# The ESP32 board will make requests to this endpoint as many times as it can per second
# This allows the ESP32 board to send OBD data to the server in real-time
@realtime_obd_stream_handling.route("/receive_obd_data", methods=["POST"])
def receive_obd_data():
    """
    Endpoint to receive OBD data from the ESP32 board in real time.

    This route:
      - Verifies that a drive session is active.
      - Retrieves the JSON OBD data payload.
      - Checks for any missing fields (none are required currently).
      - Passes the data to process_obd_data() for processing.
      - Logs and returns an appropriate HTTP response.
    """
    global obd_stream_data_buffer, current_session_id

    if not current_session_id:
        logger.error("OBD_STREAM: Data Ignored, No active session found")
        return make_response("No active session found", 400)

    obd_data = request.get_json() or {}
    required_obd_fields = []  # No mandatory fields defined for now.
    missing_obd_fields = [
        field for field in required_obd_fields if field not in obd_data
    ]
    if missing_obd_fields:
        logger.error(
            f"OBD_STREAM: Data Ignored, Missing required fields: {missing_obd_fields}"
        )
        return make_response(
            f"Data Ignored, Missing required fields: {missing_obd_fields}", 400
        )

    # Process the received OBD data.
    process_obd_data(obd_data)
    logger.debug(f"OBD_STREAM: Data received and processed: {obd_data}")
    return make_response("Data received and processed", 200)


@realtime_obd_stream_handling.route("/obd_stream_view")
def obd_stream_view():
    global obd_stream_data_buffer

    def generate():
        while True:
            try:
                while not obd_stream_data_buffer.empty():
                    obd_data = obd_stream_data_buffer.get_nowait()
                    logger.debug(f"Streamed OBD data: {obd_data}")
                    yield f"data: {json.dumps(obd_data)}\n\n"
            except queue.Empty:
                pass
            time.sleep(0.05)

    response = Response(generate(), mimetype="text/event-stream")
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Cache-Control", "no-cache")
    response.headers.add("Connection", "keep-alive")
    return response


######################################################################################
######################################################################################
##### Start and Stop Drive Session Code Below ########################################
######################################################################################
######################################################################################


# Start both stream processing functions as part of the same session and set the global session variables
# to the new session ID and name.
@realtime_camera_stream_handling.route("/start_drive_session", methods=["POST"])
def start_drive_session():
    global body_processing_thread, face_processing_thread, current_session_id, current_session_name
    global frame_count_obd, last_second_obd

    # Check if session is in progress already
    if current_session_id:
        logger.info(f"Drive session {current_session_id} already in progress")
        return make_response(
            f"Drive session {current_session_id} already in progress. Please stop it using POST /stop_drive_session before starting a new one.",
            409,
        )
    # Check if face or body threads are already running
    if body_processing_thread and body_processing_thread.is_alive():
        logger.info(f"Drive session {current_session_id} already in progress")
        return make_response(
            f"Drive session {current_session_id} already in progress. Please stop it using POST /stop_drive_session before starting a new one. Body stream is currently being processed.",
            409,
        )
    if face_processing_thread and face_processing_thread.is_alive():
        logger.info(f"Drive session {current_session_id} already in progress")
        return make_response(
            f"Drive session {current_session_id} already in progress. Please stop it using POST /stop_drive_session before starting a new one. Face stream is currently being processed.",
            409,
        )

    # Get JSON data from the POST request
    data = request.get_json() or {}
    session_name = data.get("session_name", None)

    # Generate a new session ID and assign global session variables
    sessionId = uuid.uuid4()
    current_session_id = sessionId
    current_session_name = session_name

    # Reset the frame count obd and last second obd
    frame_count_obd = 0
    last_second_obd = None

    # 1) Resolve face camera
    face_ip = resolve_mdns_via_zeroconf(FACE_STREAM_MDNS_NAME)
    if face_ip:
        face_url = f"http://{face_ip}/stream"
    else:
        # fallback to .local
        face_url = f"http://{FACE_STREAM_MDNS_NAME}/stream"

    # 2) Resolve body camera
    body_ip = resolve_mdns_via_zeroconf(BODY_STREAM_MDNS_NAME)
    if body_ip:
        body_url = f"http://{body_ip}/stream"
    else:
        body_url = f"http://{BODY_STREAM_MDNS_NAME}/stream"

    logger.info(f"Starting both: FACE => {face_url}, BODY => {body_url}")

    # 3) Create threads using the local URLs
    body_processing_thread = threading.Thread(
        target=process_stream_body, args=(body_url,)
    )
    body_processing_thread.start()

    face_processing_thread = threading.Thread(
        target=process_stream_face, args=(face_url,)
    )
    face_processing_thread.start()
    logger.info(
        f"Successfully started face and body stream processing threads and new drive session with ID: {current_session_id} and name {current_session_name}"
    )

    return make_response(
        f"New Drive session started with ID: {current_session_id} and name {current_session_name}",
        200,
    )


# Attempt to stop the entire drive session by stopping both stream processing functions and
# resetting the global session variables so the OBD data is ignored
@realtime_camera_stream_handling.route("/stop_drive_session", methods=["POST"])
def stop_drive_session():
    global body_processing_thread, body_thread_kill, face_processing_thread, face_thread_kill
    global current_session_id, current_session_name
    global frame_count_obd, last_second_obd

    if not current_session_id:
        logger.info("No active drive session to stop")
        return make_response("No active drive session to stop", 400)

    logger.debug("Starting to stop both processing threads")
    errors = []
    success_messages = []

    # Stop body processing thread
    try:
        if body_processing_thread:
            if body_processing_thread.is_alive():
                logger.debug("Stopping body processing thread")
                body_thread_kill = True
                body_processing_thread.join(timeout=5.0)

                if body_processing_thread.is_alive():
                    errors.append("Failed to stop body thread within timeout")
                    logger.error("Failed to stop body thread within timeout")
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
        logger.error(f"Error stopping body processing thread: {str(e)}")
        errors.append(f"Error stopping body thread: {str(e)}")

    # Stop face processing thread
    try:
        if face_processing_thread:
            if face_processing_thread.is_alive():
                logger.debug("Stopping face processing thread")
                face_thread_kill = True
                face_processing_thread.join(timeout=5.0)

                if face_processing_thread.is_alive():
                    errors.append("Failed to stop face thread within timeout")
                    logger.error("Failed to stop face thread within timeout")
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
        logger.error(f"Error stopping face processing thread: {str(e)}")
        errors.append(f"Error stopping face thread: {str(e)}")

    # Always reset the drive session global variables
    # This is equivalent to terminating a session
    # The OBD data will be ignored if the session is not active
    current_session_id = None
    current_session_name = None

    # Reset the frame count obd and last second obd
    frame_count_obd = 0
    last_second_obd = None

    # Handle response
    if errors:
        status_code = 500 if len(errors) == 2 else 207
        logger.error(f"Errors stopping threads: {errors}")
        return make_response(
            {"errors": errors, "successes": success_messages}, status_code
        )

    if not success_messages:
        return make_response("No processing threads were running", 200)

    logger.info("Successfully stopped all threads")
    return make_response(
        {"message": "All streams stopped successfully", "details": success_messages},
        200,
    )
