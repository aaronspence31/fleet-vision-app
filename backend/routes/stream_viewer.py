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
from firestore import body_drive_sessions, face_drive_sessions
from flask import Blueprint, Response, make_response, request
from openvino.runtime import Core
import dlib
from imutils import face_utils  # Helps convert dlib shapes to NumPy arrays

stream_viewer = Blueprint("stream_viewer", __name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Buffer for OBD data
obd_data_buffer = Queue(maxsize=1)

# Constants
# If your FACE_STREAM_URL and BODY_STREAM_URL are the same, you will get errors!
# FACE_STREAM_URL = "http://172.20.10.3/stream"  # ai thinker hotspot aaron
FACE_STREAM_URL = "http://192.168.0.108/stream"  # wrover home wifi aaron
# BODY_STREAM_URL = "http://172.20.10.8/stream"  # ai thinker hotspot aaron
BODY_STREAM_URL = "http://192.168.0.105/stream"  # ai thinker home wifi aaron
# Clip constants
CLIP_MODEL_NAME = "ViT-L/14"
CLIP_INPUT_SIZE = 224
CLIP_MODEL_PATH_BODY = "dmd29_vitbl14-hypc_429_1000_ft.pkl"

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
NUM_FRAMES_BEFORE_STORE_IN_DB = 60
frame_count_face = 0
face_frame_buffer_db = []
face_processing_thread = None
face_thread_kill = False
frame_count_body = 0
body_frame_buffer_db = []
body_processing_thread = None
body_thread_kill = False

# THIS BUFFER IS FOR VIEWING THE STREAM AND PREDICTIONS LIVE
# small max length as we only want to show the most recent frames
# these queues must be thread safe as they are accessed by multiple threads
face_stream_data_buffer = Queue(maxsize=1)
body_stream_data_buffer = Queue(maxsize=1)

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
    3: "Adjusting Radio or AC",
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

# Initialize OpenVINO model
ov_core = Core()
# Try GPU first, fall back to CPU if not available
try:
    vino_device = "GPU"
    # Test if GPU is available
    if "GPU" not in ov_core.available_devices:
        logger.warning("GPU device not available, falling back to CPU")
        vino_device = "CPU"

    logger.info(f"Using OpenVINO device: {vino_device}")
    eye_model = ov_core.compile_model(
        os.path.join(
            model_dir, "open-closed-eye-0001/open-closed-eye.onnx"
        ),  # Path to converted model
        vino_device,
    )
    logger.info("All OpenVINO models loaded successfully")
except Exception as e:
    logger.error(f"Error initializing OpenVINO models: {str(e)}")
    raise


def save_face_frames_to_firestore(sessionId):
    global face_batch_start, face_frame_buffer_db

    if len(face_frame_buffer_db) >= NUM_FRAMES_BEFORE_STORE_IN_DB:
        logger.info("FACE_STREAM: Saving face frames to Firestore")
        frame_data = list(face_frame_buffer_db)

        # Create a session document
        session_data = {
            "timestamp_start": face_batch_start,
            "timestamp_end": int(time.time()),
            "frame_count": len(frame_data),
            "frames": frame_data,
            "session_id": str(sessionId),
        }

        # Time the Firestore operation
        db_start_time = time.time()
        face_drive_sessions.add(session_data)
        db_time = (time.time() - db_start_time) * 1000  # Convert to milliseconds

        logger.info(f"FACE_STREAM: Database save completed in {db_time:.2f}ms")

        # Clear the buffer
        face_frame_buffer_db = []
        face_batch_start = None


def save_body_frames_to_firestore(sessionId):
    global body_batch_start, body_frame_buffer_db

    if len(body_frame_buffer_db) >= NUM_FRAMES_BEFORE_STORE_IN_DB:
        logger.info("BODY_STREAM: Saving body frames to Firestore")
        frame_data = list(body_frame_buffer_db)

        # Create a session document
        session_data = {
            "timestamp_start": body_batch_start,
            "timestamp_end": int(time.time()),
            "frame_count": len(frame_data),
            "frames": frame_data,
            "session_id": str(sessionId),
        }

        # Time the Firestore operation
        db_start_time = time.time()
        body_drive_sessions.add(session_data)
        db_time = (time.time() - db_start_time) * 1000  # Convert to milliseconds

        logger.info(f"BODY_STREAM: Database save completed in {db_time:.2f}ms")

        # Clear the buffer
        body_frame_buffer_db = []
        body_batch_start = None


def preprocess_frame_clip(frame, preprocess):
    # Convert BGR to RGB and to PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    # Apply CLIP preprocessing
    processed = preprocess(pil_image)
    # Add batch dimension
    return processed.unsqueeze(0)


# Detect a person in frame so we return unknown if there is no person detected
# This needs to be extremely fast as it will be called every frame
def detect_person_in_frame(frame, isFaceStream, scale_factor=1.2, min_neighbors=1):
    start_time = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if isFaceStream:
        # Face stream order: frontal face -> upper body -> profile face
        faces = faceCascade.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors
        )
        if len(faces) > 0:
            detection_time = (time.time() - start_time) * 1000
            logger.info(
                f"FACE_STREAM: Person detection completed in {detection_time:.2f}ms (frontal face detected)"
            )
            return True

        upper_bodies = upperBodyCascade.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors
        )
        if len(upper_bodies) > 0:
            detection_time = (time.time() - start_time) * 1000
            logger.info(
                f"FACE_STREAM: Person detection completed in {detection_time:.2f}ms (upper body detected)"
            )
            return True

        profile_faces = profileFaceCascade.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors
        )
        detection_time = (time.time() - start_time) * 1000
        logger.info(
            f"FACE_STREAM: Person detection completed in {detection_time:.2f}ms (Person {'detected' if len(profile_faces) > 0 else 'not detected'})"
        )
        return len(profile_faces) > 0

    else:
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


# Expect Eyes Open, Eyes Closed or Unknown as predictions on each frame
def process_stream_face(url, sessionId):
    global frame_count_face, face_stream_data_buffer, face_thread_kill, face_frame_buffer_db, face_batch_start
    EYE_MODEL_THRESHOLD = 0.5

    # Create inference request for eye model
    eye_req = eye_model.create_infer_request()

    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Limit buffer size for real-time streaming

    while not face_thread_kill:
        success, frame = cap.read()
        if not success:
            logger.error(f"FACE_STREAM: Failed to read frame from {url}")
            time.sleep(0.1)
            continue

        frame_count_face += 1
        frame_start_time = time.time()

        # Convert to grayscale for detection and landmark prediction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.15, 2)
        frame_with_boxes = frame.copy()

        # Default prediction
        classification = "Unknown"
        final_classification_prob_value = None
        eye_detected = False

        if len(faces) > 0:
            # Use the first detected face
            fx, fy, fw, fh = faces[0]
            cv2.rectangle(
                frame_with_boxes, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2
            )
            # Define roi_color as the face region (for drawing purposes)
            roi_color = frame_with_boxes[fy : fy + fh, fx : fx + fw]

            # Create a dlib rectangle using absolute face coordinates
            dlib_rect = dlib.rectangle(int(fx), int(fy), int(fx + fw), int(fy + fh))
            # Get the facial landmarks
            shape = landmark_predictor(gray, dlib_rect)
            shape_np = face_utils.shape_to_np(shape)

            # Extract landmarks for left eye (points 36–41)
            left_eye_pts = shape_np[36:42]
            # Compute bounding box for the left eye in absolute coordinates
            (ex, ey, ew, eh) = cv2.boundingRect(left_eye_pts)
            # Define separate padding factors
            padding_left = int(0.6 * ew)
            padding_right = int(0.6 * ew)
            padding_top = int(2.0 * eh)  # More padding on top
            padding_bottom = int(0.6 * eh)  # Less padding on bottom
            # Update the coordinates and dimensions with the new paddings
            ex = max(ex - padding_left, 0)
            ey = max(ey - padding_top, 0)
            ew = ew + padding_left + padding_right
            eh = eh + padding_top + padding_bottom
            # Crop the eye ROI from the full frame
            eye_roi = frame[ey : ey + eh, ex : ex + ew]
            eye_detected = True
            eye_type = "Left"

            # Draw the left eye landmarks on the full frame (for debugging)
            for x, y in left_eye_pts:
                cv2.circle(frame_with_boxes, (x, y), 1, (0, 0, 255), -1)

            # Convert absolute coordinates to relative ones for drawing on roi_color
            rel_ex = ex - fx
            rel_ey = ey - fy
            cv2.rectangle(
                roi_color, (rel_ex, rel_ey), (rel_ex + ew, rel_ey + eh), (0, 255, 0), 2
            )
            cv2.putText(
                roi_color,
                f"{eye_type}",
                (rel_ex, rel_ey - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            if eye_detected:
                # Preprocess the eye ROI for the model:
                # 1. Resize to 32x32.
                processed_eye = cv2.resize(eye_roi, (32, 32))
                # 2. Convert to float32.
                processed_eye = processed_eye.astype(np.float32)
                # 3. Normalize: subtract mean [127.0, 127.0, 127.0] and scale by 1/255.
                processed_eye = (processed_eye - 127.0) / 255.0
                # 4. Change layout from HWC to CHW.
                processed_eye = processed_eye.transpose(2, 0, 1)
                # 5. Add a batch dimension to obtain shape [1, 3, 32, 32].
                processed_eye = processed_eye[None, ...]

                # Run inference
                eye_req.infer({0: processed_eye})
                probs = eye_req.get_output_tensor().data[0]
                prob_value = float(probs[0])
                # Use standard threshold to decide eye state
                eye_state = "Open" if prob_value <= EYE_MODEL_THRESHOLD else "Closed"
                classification = f"Eyes {eye_state}"

                # Map the probability to a value between 0 and 1 for logging/display
                if prob_value <= EYE_MODEL_THRESHOLD:
                    final_classification_prob_value = 1.0 - (prob_value * 2)
                else:
                    final_classification_prob_value = (prob_value - 0.5) * 2

                logger.debug(
                    f"FACE_STREAM: {eye_type} eye classified as {eye_state} (prob: {prob_value:.2f})"
                )

        # Log processing time
        frame_time = (time.time() - frame_start_time) * 1000
        logger.info(
            f"FACE_STREAM: Frame {frame_count_face} processed in {frame_time:.2f}ms - Classification: {classification}"
        )

        # Encode frame for streaming
        _, buffer = cv2.imencode(".jpg", frame_with_boxes)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        # Firestore DB save
        if len(face_frame_buffer_db) == 0:
            face_batch_start = int(time.time())
        if len(face_frame_buffer_db) < NUM_FRAMES_BEFORE_STORE_IN_DB:
            face_frame_buffer_db.append(classification)
        if len(face_frame_buffer_db) >= NUM_FRAMES_BEFORE_STORE_IN_DB:
            save_face_frames_to_firestore(sessionId)

        # Stream output
        if face_stream_data_buffer.full():
            face_stream_data_buffer.get_nowait()
        face_stream_data_buffer.put(
            {
                "image": frame_base64,
                "prediction": classification,
                "probability": final_classification_prob_value,
                "frame_number": frame_count_face,
                "timestamp": int(time.time()),
            }
        )


# Expect body_stream_index_to_label classifications and Unknown as predictions on each frame
def process_stream_body(url, sessionId):
    global frame_count_body, clip_model, clip_preprocess, clip_classifier_body, body_batch_start, body_stream_data_buffer, body_frame_buffer_db, body_thread_kill

    if not all([clip_model, clip_preprocess, clip_classifier_body]):
        logger.error("BODY_STREAM: CLIP components not initialized")
        return

    cap = cv2.VideoCapture(url)
    cap.set(
        cv2.CAP_PROP_BUFFERSIZE, 1
    )  # limit buffer size to make stream more real-time

    while not body_thread_kill:
        success, frame = cap.read()

        if not success:
            logger.error(f"BODY_STREAM: Failed to read frame from {url}")
            time.sleep(1.0)
            continue

        frame_count_body += 1
        prediction = None
        prob_score = None
        prediction_label = "Unknown"  # Initialize at start of loop

        # person_detected = detect_person_in_frame(frame, False)
        person_detected = True  # Not using person detection for now

        if person_detected:
            try:
                prediction_start_time = time.time()
                with torch.no_grad():
                    processed = preprocess_frame_clip(frame, clip_preprocess).to(device)
                    features = clip_model.encode_image(processed)
                    features = features.cpu().numpy()

                    prediction = int(clip_classifier_body.predict(features)[0])
                    prob_score = clip_classifier_body.predict_proba(features)[0][
                        prediction
                    ]
                    prediction_label = body_stream_index_to_label.get(
                        prediction, "Unknown"
                    )

                prediction_time = (time.time() - prediction_start_time) * 1000
                logger.info(
                    f"BODY_STREAM: Frame {frame_count_body}: Processing time={prediction_time:.2f}ms, Prediction={prediction_label}, Probability={prob_score}"
                )

            except Exception as e:
                logger.error(f"BODY_STREAM: Error in inference: {str(e)}")
                prediction_label = "Unknown"
                prob_score = None
        else:
            logger.info(
                f"BODY_STREAM: Frame {frame_count_body}: No person detected, prediction set to Unknown"
            )

        # Firestore DB save
        if len(body_frame_buffer_db) == 0:
            body_batch_start = int(time.time())
        if len(body_frame_buffer_db) < NUM_FRAMES_BEFORE_STORE_IN_DB:
            body_frame_buffer_db.append(prediction_label)
        if len(body_frame_buffer_db) >= NUM_FRAMES_BEFORE_STORE_IN_DB:
            save_body_frames_to_firestore(sessionId)

        # Encode frame for streaming
        _, buffer = cv2.imencode(".jpg", frame)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        if body_stream_data_buffer.full():
            body_stream_data_buffer.get_nowait()
        body_stream_data_buffer.put(
            {
                "image": frame_base64,
                "prediction": prediction_label,
                "probability": prob_score,
                "frame_number": frame_count_body,
                "timestamp": int(time.time()),
            }
        )


# PUT ALL ROUTES BELOW -----------------------------------------------------------------------


# Start both streams as part of the same session
# Important because we need both streams to have the same session ID
@stream_viewer.route("/both_streams_start")
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
    global obd_data_buffer
    try:
        data = request.get_json()
        required_fields = ["speed", "rpm", "timestamp"]
        if all(field in data for field in required_fields):
            if obd_data_buffer.full():
                obd_data_buffer.get_nowait()
            obd_data_buffer.put(
                {
                    "speed": data["speed"],
                    "rpm": data["rpm"],
                    "timestamp": data["timestamp"],
                    # 'engine_rpm': data['engine_rpm'],
                    # 'vehicle_speed': data['vehicle_speed'],
                    # 'coolant_temp': data['coolant_temp'],
                    # 'fuel_status': data['fuel_status'],
                    # 'intake_air_temp': data['intake_air_temp'],
                    # 'mass_air_flow': data['mass_air_flow'],
                    # 'engine_load': data['engine_load'],
                    # 'fuel_rail_pressure': data['fuel_rail_pressure'],
                    # 'fuel_rail_gauge': data['fuel_rail_gauge'],
                    # 'fuel_level_input': data['fuel_level_input'],
                    # 'barometric_pressure': data['barometric_pressure'],
                    # 'distance_dtc': data['distance_dtc'],
                    # 'distance_mil': data['distance_mil'],
                    # 'trans_fluid_temp': data['trans_fluid_temp'],
                    # 'trans_gear_ratio': data['trans_gear_ratio'],
                    # 'wheel_speed': data['wheel_speed'],
                    # 'security_status': data['security_status']
                }
            )
            return make_response("Data received", 200)
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
