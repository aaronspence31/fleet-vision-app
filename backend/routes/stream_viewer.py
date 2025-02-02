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
from PIL import Image
from collections import deque
from firestore import body_drive_sessions, face_drive_sessions
from helpers.model import classify_main_batch
from tensorflow.keras.models import load_model
from flask import Blueprint, render_template, Response, make_response

stream_viewer = Blueprint("stream_viewer", __name__)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
# FACE_STREAM_URL = "http://172.20.10.6/stream"  # ai thinker hotspot aaron
# FACE_STREAM_URL = "http://192.168.0.102/stream"  # ai thinker home wifi aaron
# FACE_STREAM_URL = "http://192.168.0.110/stream"  # wrover home wifi aaron
# FACE_STREAM_URL = "http://172.20.10.4/stream"  # wrover hotspot aaron
FACE_STREAM_URL = "http://172.20.10.3/stream"
BODY_STREAM_URL = "http://172.20.10.3/stream"  # ai thinker hotspot aaron
# BODY_STREAM_URL = "http://192.168.0.104/stream"  # ai thinker home wifi aaron
# BODY_STREAM_URL = "http://192.168.0.109/stream"  # wrover home wifi aaron
# BODY_STREAM_URL = "http://172.20.10.5/stream"  # wrover hotspot aaron
BATCH_SIZE_DISTRACTION = 5
BATCH_SIZE_EYES_STATE = 5
# Might be good to not process every frame, also investigate
# if possible to not use batches at all when we skip enough frames
DISTRACTION_PROCESS_INTERVAL = 2
EYES_STATE_PROCESS_INTERVAL = 2
BINARY_EYES_STATE_PREDICTION_THRESHOLD = 0.5
BINARY_DISTRACTION_THRESHOLD = 0.3
EVENT_BATCH_SIZE_DISTRACTION = 40
EVENT_BATCH_SIZE_EYES_STATE = 40
# Clip constants
CLIP_MODEL_NAME = "ViT-L/14"
CLIP_INPUT_SIZE = 224
CLIP_PROCESS_INTERVAL = 5  # Process every 5th frame
CLIP_MODEL_PATH = "dmd29_vitbl14-hypc_429_1000_ft.pkl"

# Setup cv2 classifiers to detect eyes and faces
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
)
leftEyeCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml"
)
rightEyeCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_righteye_2splits.xml"
)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, "models")

frame_count_face = 0
frame_count_body = 0

body_frame_buffer = deque(maxlen=60)
body_buffer_lock = threading.Lock()
body_processing_thread = None

face_frame_buffer = deque(maxlen=60)
face_buffer_lock = threading.Lock()
face_processing_thread = None

# Load models and label mappings
try:
    binary_eyes_state_model = load_model(
        os.path.join(
            model_dir,
            "binary_eyes_CustomCNN_16batchsize_0.001learningrate_100epochs",
        )
    )
    binary_distraction_model = load_model(
        os.path.join(
            model_dir, "binary_MobileNetV3Small_32batchsize_1e-06learningrate_4epochs"
        )
    )

    with open(
        os.path.join(model_dir, "label_mapping_binary_eyes_state.json"), "r"
    ) as f:
        eyes_label_to_index = json.load(f)
    with open(
        os.path.join(model_dir, "label_mapping_binary_distraction.json"), "r"
    ) as f:
        distraction_label_to_index = json.load(f)

    eyes_index_to_label = {int(v): k for k, v in eyes_label_to_index.items()}
    distraction_index_to_label = {
        int(v): k for k, v in distraction_label_to_index.items()
    }

    logger.info("Models and label mappings loaded successfully")
except Exception as e:
    logger.error(f"Error loading models or label mappings: {str(e)}")
    raise

# Add global variables for clip model body stream processing
clip_model = None
clip_preprocess = None
clip_classifier = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Clip label mappings
clip_index_to_label = {
    0: "drinking",
    1: "hair_and_makeup",
    2: "phonecall_right",
    3: "radio",
    4: "reach_backseat",
    5: "reach_side",
    6: "safe_drive",
    7: "talking_to_passenger",
    8: "texting_right",
    9: "yawning",
}


# Initialize CLIP components
def init_clip():
    global clip_model, clip_preprocess, clip_classifier
    clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device)
    clip_model.eval()
    clip_classifier = joblib.load(os.path.join(model_dir, CLIP_MODEL_PATH))


# load clip stuff
try:
    init_clip()
    logger.info("CLIP model and classifier loaded successfully")
except Exception as e:
    logger.error(f"Error loading CLIP model: {str(e)}")


def save_frames_to_firestore(sessionId):
    """
    Save accumulated frames to Firestore when buffer is full
    """
    global batch_start

    with body_buffer_lock:
        if len(body_frame_buffer) >= 60:
            frame_data = list(body_frame_buffer)

            # Create a session document
            session_data = {
                "timestamp_start": batch_start,
                "timestamp_end": int(time.time()),
                "frame_count": len(frame_data),
                "frames": frame_data,
                "session_id": str(sessionId),
            }

            # Add to Firestore
            body_drive_sessions.add(session_data)

            # Clear the buffer
            body_frame_buffer.clear()
            batch_start = None


def save_face_frames_to_firestore(sessionId):
    global face_batch_start

    with body_buffer_lock:
        if len(body_frame_buffer) >= 12:
            frame_data = list(body_frame_buffer)

            # Create a session document
            session_data = {
                "timestamp_start": face_batch_start,
                "timestamp_end": int(time.time()),
                "frame_count": len(frame_data),
                "frames": frame_data,
                "session_id": str(sessionId),
            }

            # Add to Firestore
            face_drive_sessions.add(session_data)

            # Clear the buffer
            body_frame_buffer.clear()
            face_batch_start = None


# def preprocess_image(frame, target_size=(224, 224)):
#     rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     resized_rgb_img = cv2.resize(rgb_img, target_size, interpolation=cv2.INTER_AREA)
#     normalized_rgb_img = resized_rgb_img.astype(np.float32) / 255.0
#     return normalized_rgb_img


# the image will already come in in grayscale as the model expects
def preprocess_image_face(frame, target_size=(32, 32)):
    # rgb_image = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    # rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_grayscale_img = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    normalized_grayscale_img = resized_grayscale_img.astype(np.float32) / 255.0
    return normalized_grayscale_img


def preprocess_frame_clip(frame, preprocess):
    """
    Preprocess frame using CLIP's standard preprocessing pipeline
    - Convert BGR to RGB
    - Convert to PIL Image
    - Apply CLIP preprocessing (resizing, normalization)
    """
    # Convert BGR to RGB and to PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Apply CLIP preprocessing
    processed = preprocess(pil_image)

    # Add batch dimension
    return processed.unsqueeze(0)


def predict_batch(
    batch,
    model,
    index_to_label,
    is_distraction_model=False,
    curBatchSize=BATCH_SIZE_EYES_STATE,
):
    if is_distraction_model:
        predictions = model.predict(np.array(batch), batch_size=BATCH_SIZE_DISTRACTION)
    else:
        predictions = model.predict(np.array(batch), batch_size=curBatchSize)
    predicted_labels = []
    for pred in predictions:
        if is_distraction_model:
            if pred[0] >= BINARY_DISTRACTION_THRESHOLD:
                predicted_labels.append(index_to_label[1])  # Safe driving
            else:
                predicted_labels.append(index_to_label[0])  # Distracted
        else:
            if pred[0] >= BINARY_EYES_STATE_PREDICTION_THRESHOLD:
                predicted_labels.append(index_to_label[1])  # Eyes open
            else:
                predicted_labels.append(index_to_label[0])  # Eyes closed
    return predicted_labels


# def process_stream(stream_url, model, index_to_label, is_distraction_model=False):
#     global frame_count_body
#     cap = cv2.VideoCapture(stream_url)
#     currBufferSize = 0
#     predictions_buffer = []
#     prevEvent = {}
#     while True:
#         batch_frames = []
#         batch_start_frame_count = frame_count_body
#         for _ in range(BATCH_SIZE_DISTRACTION):
#             ret, frame = cap.read()
#             if not ret:
#                 logger.error(f"Failed to read frame from {stream_url}")
#                 time.sleep(0.1)
#                 continue
#             batch_frames.append(frame)
#             frame_count_body += 1
#             currBufferSize += 1

#         if len(batch_frames) == BATCH_SIZE_DISTRACTION:
#             processed_batch = [preprocess_image(frame) for frame in batch_frames]
#             predictions = predict_batch(
#                 processed_batch, model, index_to_label, is_distraction_model
#             )
#             predictions_buffer += predictions

#             event = 0

#             if currBufferSize >= EVENT_BATCH_SIZE_DISTRACTION:
#                 event_label = classify_main_batch(predictions_buffer)

#                 cont = (
#                     1
#                     if "label" in prevEvent and prevEvent["label"] == event_label
#                     else 0
#                 )
#                 event = {
#                     "frameStart": frame_count_body - EVENT_BATCH_SIZE_DISTRACTION,
#                     "frameEnd": frame_count_body,
#                     "label": event_label,
#                     "cont": cont,
#                 }
#                 prevEvent = event

#                 predictions_buffer = []
#                 currBufferSize = 0

#             middle_frame = batch_frames[BATCH_SIZE_DISTRACTION // 2]
#             _, buffer = cv2.imencode(".jpg", middle_frame)
#             frame_base64 = base64.b64encode(buffer).decode("utf-8")

#             yield (
#                 f"data: {{\n"
#                 f'data: "image": "{frame_base64}",\n'
#                 f'data: "event": "{event}",\n'
#                 f'data: "first_frame_num": "{batch_start_frame_count + 1}",\n'
#                 f'data: "predictions": {json.dumps(predictions)}\n'
#                 f"data: }}\n\n"
#             )


# process the face stream
# use cv2 haarcascade classifier to detect eyes
# then use the binary_eyes_state_model to predict the state of each eye that was detected
def process_stream_face(stream_url, model, index_to_label, sessionId):
    global frame_count_face, face_batch_start
    cap = cv2.VideoCapture(stream_url)
    currBufferSize = 0
    predictions_buffer = []
    prevEvent = {}
    while True:
        batch_frames = []
        batch_start_frame_count = frame_count_face
        while len(batch_frames) < BATCH_SIZE_EYES_STATE:
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Failed to read frame from {stream_url}")
                time.sleep(0.1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.15, 2)

            frame_with_boxes = frame.copy()

            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]  # Get the first face
                cv2.rectangle(
                    frame_with_boxes, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2
                )

                roi_gray = gray[fy : fy + fh, fx : fx + fw]
                roi_color = frame_with_boxes[fy : fy + fh, fx : fx + fw]

                # Detect left and right eyes separately
                left_eye = leftEyeCascade.detectMultiScale(roi_gray, 1.05, 5)
                right_eye = rightEyeCascade.detectMultiScale(roi_gray, 1.05, 5)

                eye_images = []

                # Process left eye
                if len(left_eye) > 0:
                    ex, ey, ew, eh = left_eye[0]
                    cv2.rectangle(
                        roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1
                    )
                    eye_roi = roi_gray[ey : ey + eh, ex : ex + ew]
                    eye_images.append(eye_roi)
                else:
                    eye_images.append(None)

                # Process right eye
                if len(right_eye) > 0:
                    ex, ey, ew, eh = right_eye[0]
                    cv2.rectangle(
                        roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1
                    )
                    eye_roi = roi_gray[ey : ey + eh, ex : ex + ew]
                    eye_images.append(eye_roi)
                else:
                    eye_images.append(None)

                batch_frames.append((frame_with_boxes, eye_images[0], eye_images[1]))
                frame_count_face += 1
                currBufferSize += 1
            else:
                batch_frames.append((frame_with_boxes, None, None))
                frame_count_face += 1
                currBufferSize += 1

        if len(batch_frames) == BATCH_SIZE_EYES_STATE:
            # Process the batch
            processed_batch = []
            # we need to keep track of the frames that didn't have any eyes detected
            # so that we can insert unknown predictions at the appropriate indices for them
            unknown_frame_indices = []
            curr_index = 0
            num_images_in_model_batch = 0
            for frame_data in batch_frames:
                # in this case, we detected two eyes
                if frame_data[1] is not None and frame_data[2] is not None:
                    # for now we will just process and feed the left eye into the model and ignore the other one
                    # thus we are only making predictions on the overall eyes state based on a single eye
                    # this should work fine for now and it simplifies this code greatly
                    processed_batch.append(
                        preprocess_image_face(frame_data[1], (32, 32))
                    )
                    num_images_in_model_batch += 1

                # in this case, we detected one eye
                elif frame_data[1] is not None and frame_data[2] is None:
                    processed_batch.append(
                        preprocess_image_face(frame_data[1], (32, 32))
                    )
                    num_images_in_model_batch += 1

                # in this case, we detected no eyes
                # no need to preprocess as we have nothing to feed to the model
                else:
                    unknown_frame_indices.append(curr_index)
                curr_index += 1

            predictions = []
            is_distraction_model = False
            if num_images_in_model_batch > 0:
                predictions = predict_batch(
                    processed_batch,
                    model,
                    index_to_label,
                    is_distraction_model,
                    num_images_in_model_batch,
                )
                # Insert "Unknown" predictions for frames where no eyes were detected
                for idx in unknown_frame_indices:
                    predictions.insert(idx, "Unknown")
            else:  # If no eyes detected in any frame, insert "Unknown" for all
                predictions = ["Unknown"] * BATCH_SIZE_EYES_STATE

            predictions_buffer += predictions

            event = 0

            if currBufferSize >= EVENT_BATCH_SIZE_EYES_STATE:
                event_label = classify_main_batch(predictions_buffer)

                cont = (
                    1
                    if "label" in prevEvent and prevEvent["label"] == event_label
                    else 0
                )
                event = {
                    "frameStart": frame_count_face - EVENT_BATCH_SIZE_EYES_STATE,
                    "frameEnd": frame_count_face,
                    "label": event_label,
                    "cont": cont,
                }
                prevEvent = event

                predictions_buffer = []
                currBufferSize = 0

            middle_frame_data = batch_frames[BATCH_SIZE_EYES_STATE // 2]
            middle_frame, left_eye, right_eye = middle_frame_data

            middle_frame = cv2.cvtColor(middle_frame, cv2.COLOR_BGR2RGB)

            # Encode the main frame (now with bounding boxes for face and eyes)
            _, buffer = cv2.imencode(".jpg", middle_frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            with face_buffer_lock:
                # If this is the first frame in a new session
                if len(face_frame_buffer) == 0:
                    face_batch_start = int(time.time())

                face_frame_buffer.append(predictions)

            if len(face_frame_buffer) >= 12:
                save_face_frames_to_firestore(sessionId)

            yield (
                f"data: {{\n"
                f'data: "image": "{frame_base64}",\n'
                f'data: "event": "{event}",\n'
                f'data: "first_frame_num": "{batch_start_frame_count + 1}",\n'
                f'data: "predictions": {json.dumps(predictions)}\n'
                f"data: }}\n\n"
            )


def process_stream_clip(url, sessionId):
    global frame_count_body, clip_model, clip_preprocess, clip_classifier, batch_start

    if not all([clip_model, clip_preprocess, clip_classifier]):
        logger.error("CLIP components not initialized")
        return

    cap = cv2.VideoCapture(url)

    while True:
        success, frame = cap.read()
        if not success:
            logger.error(f"Failed to read frame from {url}")
            time.sleep(0.1)
            continue

        frame_count_body += 1
        prediction = None
        prob_score = None
        prediction_label = "No prediction"  # Initialize at start of loop

        if frame_count_body % CLIP_PROCESS_INTERVAL == 0:
            height, width, channels = frame.shape
            logger.info(f"Frame dimensions: {width}x{height}x{channels}")
            try:
                with torch.no_grad():
                    processed = preprocess_frame_clip(frame, clip_preprocess).to(device)
                    features = clip_model.encode_image(processed)
                    features = features.cpu().numpy()

                    prediction = int(clip_classifier.predict(features)[0])
                    prob_score = clip_classifier.predict_proba(features)[0][prediction]
                    prediction_label = clip_index_to_label.get(prediction, "Unknown")

                    with body_buffer_lock:
                        # If this is the first frame in a new session
                        if len(body_frame_buffer) == 0:
                            batch_start = int(time.time())

                        body_frame_buffer.append(prediction_label)

                    # Check if we should save to Firestore
                    if len(body_frame_buffer) >= 60:
                        save_frames_to_firestore(sessionId)

                    logger.info(
                        f"Frame {frame_count_body}: Prediction={prediction_label}, Probability={prob_score}"
                    )

            except Exception as e:
                logger.error(f"Error in inference: {str(e)}")
                prediction_label = "Error"
                prob_score = None

            # Encode frame for streaming
            _, buffer = cv2.imencode(".jpg", frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            yield (
                f"data: {{\n"
                f'data: "image": "{frame_base64}",\n'
                f'data: "prediction": "{prediction_label}",\n'
                f'data: "probability": "{prob_score}",\n'
                f'data: "frame_number": "{frame_count_body}"\n'
                f"data: }}\n\n"
            )


# @stream_viewer.route("/")
# def index():
#     return render_template("index.html")


@stream_viewer.route("/face_stream")
def face_stream():
    global face_processing_thread

    sessionId = uuid.uuid4()
    body_processing_thread = threading.Thread(
        target=process_stream_face,
        args=(FACE_STREAM_URL, binary_eyes_state_model, eyes_index_to_label, sessionId),
    )
    body_processing_thread.start()
    return make_response(f"Face processing started for session {sessionId}", 200)
    # return Response(
    #     process_stream_face(
    #         FACE_STREAM_URL, binary_eyes_state_model, eyes_index_to_label
    #     ),
    #     mimetype="text/event-stream",
    # )


@stream_viewer.route("/face_stream_stop")
def face_stream():
    return Response(
        process_stream_face(
            FACE_STREAM_URL, binary_eyes_state_model, eyes_index_to_label
        ),
        mimetype="text/event-stream",
    )


# @stream_viewer.route("/body_stream")
# def body_stream():
#     return Response(
#         process_stream(
#             BODY_STREAM_URL, binary_distraction_model, distraction_index_to_label, True
#         ),
#         mimetype="text/event-stream",
#     )


@stream_viewer.route("/body_stream_clip")
def body_stream_clip():
    global body_processing_thread

    sessionId = uuid.uuid4()
    body_processing_thread = threading.Thread(
        target=process_stream_clip, args=(BODY_STREAM_URL, sessionId)
    )
    body_processing_thread.start()
    return make_response(f"Processing started for session {sessionId}", 200)
    # return Response(
    #     process_stream_clip(BODY_STREAM_URL),
    #     mimetype="text/event-stream",
    # )


@stream_viewer.route("/body_stream_clip_stop")
def body_stream_clip():
    global body_processing_thread

    if body_processing_thread:
        body_processing_thread.join()
        body_processing_thread = None
        return make_response(f"Processing stopped", 200)
    else:
        return make_response(f"No processing thread to stop", 200)

    # return Response(
    #     process_stream_clip(BODY_STREAM_URL),
    #     mimetype="text/event-stream",
    # )


@stream_viewer.route("/stream_info", methods=["GET"])
def get_stream_info():
    return {
        "BATCH_SIZE_DISTRACTION": BATCH_SIZE_DISTRACTION,
        "BATCH_SIZE_EYES_STATE": BATCH_SIZE_EYES_STATE,
        "BINARY_EYES_STATE_PREDICTION_THRESHOLD": BINARY_EYES_STATE_PREDICTION_THRESHOLD,
        "BINARY_DISTRACTION_THRESHOLD": BINARY_DISTRACTION_THRESHOLD,
    }
