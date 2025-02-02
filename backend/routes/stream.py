# from flask import Blueprint
# from tensorflow.keras.models import load_model
# import numpy as np
# import cv2
# import os
# import json
# import logging
# import time
# import threading
# from queue import Queue, Empty

# stream = Blueprint("stream", __name__)

# # Setup logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
# logger = logging.getLogger(__name__)

# # Constants
# BATCH_SIZE = 16
# QUEUE_SIZE = 1000

# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# model_dir = os.path.join(base_dir, "models")

# # Load models and label mappings
# try:
#     eyes_model = load_model(
#         os.path.join(model_dir, "face_eyes_state_MobileNetV3Small_16_30epochs")
#     )
#     actions_model = load_model(
#         os.path.join(model_dir, "face_eyes_state_MobileNetV3Small_16_30epochs")
#     )

#     with open(os.path.join(model_dir, "label_mapping_face_eyes_state.json"), "r") as f:
#         eyes_label_to_index = json.load(f)
#     with open(os.path.join(model_dir, "label_mapping_face_eyes_state.json"), "r") as f:
#         actions_label_to_index = json.load(f)

#     eyes_index_to_label = {int(v): k for k, v in eyes_label_to_index.items()}
#     actions_index_to_label = {int(v): k for k, v in actions_label_to_index.items()}

#     logger.info("Models and label mappings loaded successfully")
# except Exception as e:
#     logger.error(f"Error loading models or label mappings: {str(e)}")
#     raise

# # Queue for face frames
# face_queue = Queue(maxsize=QUEUE_SIZE)


# def preprocess_image(frame, target_size=(224, 224)):
#     rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     resized_rgb_img = cv2.resize(rgb_img, target_size, interpolation=cv2.INTER_AREA)
#     normalized_rgb_img = resized_rgb_img.astype(np.float32) / 255.0
#     return normalized_rgb_img


# def predict_batch(batch, model, index_to_label):
#     predictions = model.predict(np.array(batch))
#     predicted_labels = [index_to_label[np.argmax(pred)] for pred in predictions]
#     return predicted_labels


# def process_stream(stream_url, queue):
#     while True:
#         cap = cv2.VideoCapture(stream_url)
#         cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
#         if not cap.isOpened():
#             logger.error(f"Failed to open stream: {stream_url}")
#             time.sleep(5)  # Wait before retrying
#             continue

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 logger.error(f"Failed to read frame from {stream_url}")
#                 break

#             processed_frame = preprocess_image(frame)

#             if not queue.full():
#                 queue.put(processed_frame)
#             else:
#                 logger.warning("Queue is full, dropping frame")

#             logger.info(f"Current queue size: {queue.qsize()}")

#         cap.release()
#         time.sleep(5)  # Wait before reconnecting


# def process_model_queue(
#     queue, eyes_model, actions_model, eyes_index_to_label, actions_index_to_label
# ):
#     batch = []
#     while True:
#         try:
#             frame = queue.get(timeout=1)
#             batch.append(frame)

#             if len(batch) == BATCH_SIZE:
#                 start_time = time.time()
#                 eyes_predictions = predict_batch(batch, eyes_model, eyes_index_to_label)
#                 actions_predictions = predict_batch(
#                     batch, actions_model, actions_index_to_label
#                 )
#                 end_time = time.time()

#                 logger.info(f"Eyes predictions: {eyes_predictions}")
#                 logger.info(f"Actions predictions: {actions_predictions}")
#                 logger.info(f"Prediction time: {end_time - start_time:.4f} seconds")
#                 logger.info(f"Current queue size: {queue.qsize()}")

#                 batch = []
#         except Empty:
#             if batch:
#                 start_time = time.time()
#                 eyes_predictions = predict_batch(batch, eyes_model, eyes_index_to_label)
#                 actions_predictions = predict_batch(
#                     batch, actions_model, actions_index_to_label
#                 )
#                 end_time = time.time()

#                 logger.info(f"Eyes predictions: {eyes_predictions}")
#                 logger.info(f"Actions predictions: {actions_predictions}")
#                 logger.info(f"Prediction time: {end_time - start_time:.4f} seconds")
#                 logger.info(f"Current queue size: {queue.qsize()}")

#                 batch = []


# def start_streams():
#     face_stream_url = "http://192.168.2.175/stream"  # Adjust if needed

#     # Start stream processing thread
#     threading.Thread(
#         target=process_stream, args=(face_stream_url, face_queue), daemon=True
#     ).start()

#     # Start model processing thread
#     threading.Thread(
#         target=process_model_queue,
#         args=(
#             face_queue,
#             eyes_model,
#             actions_model,
#             eyes_index_to_label,
#             actions_index_to_label,
#         ),
#         daemon=True,
#     ).start()

#     logger.info("Streams started")


# @stream.route("/stream_info", methods=["GET"])
# def get_stream_info():
#     return {
#         "active_streams": 1,
#         "face_queue_size": face_queue.qsize(),
#         "batch_size": BATCH_SIZE,
#         "queue_size": QUEUE_SIZE,
#     }


# if __name__ == "__main__":
#     start_streams()
#     # Keep the main thread alive
#     while True:
#         time.sleep(1)
