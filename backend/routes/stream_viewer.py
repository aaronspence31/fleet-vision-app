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
from queue import Queue
from firestore import body_drive_sessions, face_drive_sessions
from helpers.model import classify_main_batch
from flask import Blueprint, Response, make_response
from transformers import ViTFeatureExtractor, ViTForImageClassification

stream_viewer = Blueprint("stream_viewer", __name__)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
# FACE_STREAM_URL = "http://172.20.10.8/stream"  # ai thinker hotspot aaron
FACE_STREAM_URL = "http://192.168.0.111/stream"  # ai thinker home wifi aaron
# FACE_STREAM_URL = "http://172.20.10.4/stream"  # wrover hotspot aaron
# BODY_STREAM_URL = "http://172.20.10.3/stream"  # ai thinker hotspot aaron
BODY_STREAM_URL = "http://192.168.0.112/stream"  # ai thinker home wifi aaron
# BODY_STREAM_URL = "http://172.20.10.5/stream"  # wrover hotspot aaron
BATCH_SIZE_EYES_STATE = 1
EYES_STATE_STREAM_PROCESS_INTERVAL = 5
# WE SHOULD RECONSIDER HOW WE DO THESE EVENTS NOW THAT WE ARENT REALLY BATCHING
EVENT_BATCH_SIZE_EYES_STATE = 5
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

# RIGHT NOW WE ONLY STORE IN DB ONCE A FULL 10 FRAMES HAS BEEN COMPLETED
# WHY BUFFER RATHER THAN JUST SAVING TO FIRESTORE IMMEDIATELY?
body_frame_buffer = []
body_processing_thread = None
face_frame_buffer = []
face_processing_thread = None

# THIS BUFFER IS FOR VIEWING THE STREAM AND PREDICTIONS LIVE
# small max length as we only want to show the most recent frames
# these queues much actually be thread safe as they are accessed by multiple threads
face_stream_data_buffer = Queue(maxsize=2)
body_stream_data_buffer = Queue(maxsize=2)

# Load eyes state model
try:
    model_name = "dima806/closed_eyes_image_detection"
    eyes_feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    eyes_model = ViTForImageClassification.from_pretrained(model_name)

    # Move the model to GPU if available
    eyesDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eyes_model.to(eyesDevice)

    logger.info("Eyes state model loaded successfully")
except Exception as e:
    logger.error(f"Error loading eyes state model: {str(e)}")
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


def save_body_frames_to_firestore(sessionId):
    global batch_start, body_frame_buffer

    if len(body_frame_buffer) >= 10:
        logger.info("Saving body frames to Firestore")
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
        body_frame_buffer = []
        batch_start = None


def save_face_frames_to_firestore(sessionId):
    global face_batch_start, face_frame_buffer

    if len(face_frame_buffer) >= 10:
        logger.info("Saving face frames to Firestore")
        frame_data = list(face_frame_buffer)

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
        face_frame_buffer = []
        face_batch_start = None


# the image will already come in in grayscale as the model expects
def preprocess_image_face(frame):
    # Convert BGR to RGB and to PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    # Apply preprocessing
    processed = eyes_feature_extractor(images=pil_image, return_tensors="pt")
    # Move the inputs to the same device as the model
    return {k: v.to(eyesDevice) for k, v in processed.items()}


def preprocess_frame_clip(frame, preprocess):
    # Convert BGR to RGB and to PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    # Apply CLIP preprocessing
    processed = preprocess(pil_image)
    # Add batch dimension
    return processed.unsqueeze(0)


# TODO: this needs to be updated to get rid of any batching and events as these are causing issues
# TODO: this also needs to be updated to use the binary eyes state model correctly if possible with this model
# process the face stream
# use cv2 haarcascade classifier to detect eyes
# then use the binary_eyes_state_model to predict the state of each eye that was detected
def process_stream_face(stream_url, sessionId):
    global frame_count_face, face_batch_start, face_stream_data_buffer, face_frame_buffer
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

            frame_count_face += 1
            if frame_count_face % EYES_STATE_STREAM_PROCESS_INTERVAL == 0:
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

                    batch_frames.append(
                        (frame_with_boxes, eye_images[0], eye_images[1])
                    )
                    currBufferSize += 1
                else:
                    batch_frames.append((frame_with_boxes, None, None))
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
                    # WE ONLY USE A SINGLE EYE FOR PREDICTIONS!!! WOULD PROBABLY BE BEST TO USE BOTH SOMEHOW
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
            if num_images_in_model_batch > 0:
                for item in processed_batch:
                    # NEW EYES OPEN CLOSED INFERENCE
                    # NEED TO DOUBLE CHECK THIS IS WORKING CORRECT
                    with torch.no_grad():
                        output = eyes_model(**item)
                    # Get predicted class index (assuming index 0 corresponds to "closed" and 1 to "open")
                    logits = output.logits
                    predicted_class_idx = logits.argmax(-1).item()
                    finalClassification = (
                        "open" if predicted_class_idx == 1 else "closed"
                    )
                    predictions.append(finalClassification)

                # Insert "Unknown" predictions for frames where no eyes were detected
                for idx in unknown_frame_indices:
                    predictions.insert(idx, "Unknown")
            else:  # If no eyes detected in any frame, insert "Unknown" for all
                predictions = ["Unknown"] * BATCH_SIZE_EYES_STATE

            logger.info(f"Made face predictions {predictions}")

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

            # Firestore
            if len(face_frame_buffer) == 0:
                face_batch_start = int(time.time())
            if len(face_frame_buffer) < 10:
                face_frame_buffer.append(predictions)
            if len(face_frame_buffer) >= 10:
                save_face_frames_to_firestore(sessionId)

            # Instead of yielding, we now add to a queue that can be read from to view the stream live
            if face_stream_data_buffer.full():
                face_stream_data_buffer.get_nowait()
            face_stream_data_buffer.put(
                {
                    "image": frame_base64,
                    "event": event,
                    "first_frame_num": batch_start_frame_count + 1,
                    "predictions": predictions,
                }
            )
            # yield (
            #     f"data: {{\n"
            #     f'data: "image": "{frame_base64}",\n'
            #     f'data: "event": "{event}",\n'
            #     f'data: "first_frame_num": "{batch_start_frame_count + 1}",\n'
            #     f'data: "predictions": {json.dumps(predictions)}\n'
            #     f"data: }}\n\n"
            # )


def process_stream_body(url, sessionId):
    global frame_count_body, clip_model, clip_preprocess, clip_classifier, batch_start, body_stream_data_buffer, body_frame_buffer

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

                # Firestore
                if len(body_frame_buffer) == 0:
                    batch_start = int(time.time())
                if len(body_frame_buffer) < 10:
                    body_frame_buffer.append(prediction_label)
                if len(body_frame_buffer) >= 10:
                    save_body_frames_to_firestore(sessionId)

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

            # Instead of yielding, we now add to a queue that can be read from to view the stream live
            if body_stream_data_buffer.full():
                body_stream_data_buffer.get_nowait()
            body_stream_data_buffer.put(
                {
                    "image": frame_base64,
                    "prediction": prediction_label,
                    "probability": prob_score,
                    "frame_number": frame_count_body,
                }
            )
            # yield (
            #     f"data: {{\n"
            #     f'data: "image": "{frame_base64}",\n'
            #     f'data: "prediction": "{prediction_label}",\n'
            #     f'data: "probability": "{prob_score}",\n'
            #     f'data: "frame_number": "{frame_count_body}"\n'
            #     f"data: }}\n\n"
            # )


# PUT ALL ROUTES BELOW -----------------------------------------------------------------------


@stream_viewer.route("/face_stream_start")
def face_stream_start():
    global face_processing_thread

    sessionId = uuid.uuid4()
    face_processing_thread = threading.Thread(
        target=process_stream_face,
        args=(FACE_STREAM_URL, sessionId),
    )
    face_processing_thread.start()
    return make_response(f"Face processing started for session {sessionId}", 200)


@stream_viewer.route("/face_stream_stop")
def face_stream_stop():
    global face_processing_thread

    if face_processing_thread:
        face_processing_thread.join()
        face_processing_thread = None
        return make_response(f"Processing stopped", 200)
    return make_response(f"No processing thread to stop", 200)


@stream_viewer.route("/face_stream_view")
def face_stream_view():
    global face_stream_data_buffer

    def generate():
        while True:
            try:
                while not face_stream_data_buffer.empty():
                    frame = face_stream_data_buffer.get_nowait()
                    yield f"data: {json.dumps(frame)}\n\n"
            except Queue.Empty:
                pass
            time.sleep(0.1)  # Prevent CPU thrashing

    return Response(generate(), mimetype="text/event-stream")


@stream_viewer.route("/body_stream_clip_start")
def body_stream_clip_start():
    global body_processing_thread

    sessionId = uuid.uuid4()
    body_processing_thread = threading.Thread(
        target=process_stream_body, args=(BODY_STREAM_URL, sessionId)
    )
    body_processing_thread.start()
    return make_response(f"Body processing started for session {sessionId}", 200)


@stream_viewer.route("/body_stream_clip_stop")
def body_stream_clip_stop():
    global body_processing_thread

    if body_processing_thread:
        body_processing_thread.join()
        body_processing_thread = None
        return make_response(f"Processing stopped", 200)
    return make_response(f"No processing thread to stop", 200)


@stream_viewer.route("/body_stream_clip_view")
def body_stream_clip_view():
    global body_stream_data_buffer

    def generate():
        while True:
            try:
                while not body_stream_data_buffer.empty():
                    frame = body_stream_data_buffer.get_nowait()
                    yield f"data: {json.dumps(frame)}\n\n"
            except Queue.Empty:
                pass
            time.sleep(0.1)  # Prevent CPU thrashing

    return Response(generate(), mimetype="text/event-stream")
