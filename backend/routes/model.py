# from flask import Blueprint, request, jsonify, current_app
# from tensorflow.keras.models import load_model
# import numpy as np
# import cv2
# import os
# import logging
# import time
# import imghdr

# model = Blueprint("model", __name__)
# PREDICTION_THRESHOLD = 0.5
# MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
# ALLOWED_IMAGE_TYPES = {'jpg', 'jpeg', 'png'}

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# model_dir = os.path.join(base_dir, 'models')

# try:
#     face_model = load_model(os.path.join(model_dir, 'saved_model_MobileNetV3Small_16_face'))
#     body_model = load_model(os.path.join(model_dir, 'saved_model_MobileNetV3Small_16_body'))

#     face_label_to_index = np.load(os.path.join(model_dir, 'face_label_mapping.npy'), allow_pickle=True).item()
#     body_label_to_index = np.load(os.path.join(model_dir, 'body_label_mapping.npy'), allow_pickle=True).item()

#     face_index_to_label = {v: k for k, v in face_label_to_index.items()}
#     body_index_to_label = {v: k for k, v in body_label_to_index.items()}

#     logger.info("Models and label mappings loaded successfully")
# except Exception as e:
#     logger.error(f"Error loading models or label mappings: {str(e)}")
#     raise

# def validate_image(image_bytes):
#     if len(image_bytes) > MAX_IMAGE_SIZE:
#         raise ValueError(f"Image size exceeds the maximum allowed size of {MAX_IMAGE_SIZE / 1024 / 1024} MB")
#     file_type = imghdr.what(None, image_bytes)
#     if file_type not in ALLOWED_IMAGE_TYPES:
#         raise ValueError(f"Unsupported image type: {file_type}. Allowed types are {ALLOWED_IMAGE_TYPES}")

# def preprocess_image(image_bytes, target_size=(224, 224)):
#     try:
#         validate_image(image_bytes)
#         nparr = np.frombuffer(image_bytes, np.uint8)
#         bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         if bgr_img is None:
#             raise ValueError("Failed to decode image")
#         rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
#         resized_rgb_img = cv2.resize(rgb_img, target_size, interpolation=cv2.INTER_AREA)
#         normalized_rgb_img = resized_rgb_img.astype(np.float32) / 255.0
#         return normalized_rgb_img
#     except Exception as e:
#         logger.error(f"Error in image preprocessing: {str(e)}")
#         raise

# def predict_image(img_array, model, index_to_label):
#     predictions = model.predict(img_array)
#     predicted_labels = [[index_to_label[i] for i, prob in enumerate(pred) if prob > PREDICTION_THRESHOLD] for pred in predictions]
#     return predicted_labels

# def process_single_image(file, model, index_to_label):
#     try:
#         total_start_time = time.time()
#         img_bytes = file.read()
#         logger.info(f"Read image with size {len(img_bytes)} bytes")

#         validate_image(img_bytes)
#         logger.info("Validated image successfully")

#         preprocess_start_time = time.time()
#         img_array = preprocess_image(img_bytes)
#         preprocess_end_time = time.time()
#         image_processing_time = preprocess_end_time - preprocess_start_time
#         logger.info("Processed image successfully")

#         prediction_start_time = time.time()
#         predicted_labels = predict_image(np.expand_dims(img_array, axis=0), model, index_to_label)[0]
#         prediction_end_time = time.time()
#         prediction_time = prediction_end_time - prediction_start_time

#         total_end_time = time.time()
#         total_processing_time = total_end_time - total_start_time

#         logger.info(f"Prediction made successfully. Labels: {predicted_labels}, "
#                     f"Image Processing Time: {image_processing_time:.4f} seconds, "
#                     f"Prediction Time: {prediction_time:.4f} seconds, "
#                     f"Total Time: {total_processing_time:.4f} seconds")

#         return {
#             "predictions": predicted_labels,
#             "image_processing_time": image_processing_time,
#             "prediction_time": prediction_time,
#             "total_processing_time": total_processing_time
#         }
#     except Exception as e:
#         logger.error(f"Error during prediction: {str(e)}", exc_info=True)
#         raise

# @model.route("/predict/face", methods=["POST"])
# def predict_face():
#     if 'image' not in request.files:
#         logger.warning("No image file provided in the request")
#         return jsonify({"error": "No image file provided"}), 400

#     file = request.files['image']
#     try:
#         result = process_single_image(file, face_model, face_index_to_label)
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @model.route("/predict/body", methods=["POST"])
# def predict_body():
#     if 'image' not in request.files:
#         logger.warning("No image file provided in the request")
#         return jsonify({"error": "No image file provided"}), 400

#     file = request.files['image']
#     try:
#         result = process_single_image(file, body_model, body_index_to_label)
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @model.route("/batchPredict/face", methods=["POST"])
# def batch_predict_face():
#     return batch_predict(face_model, face_index_to_label)

# @model.route("/batchPredict/body", methods=["POST"])
# def batch_predict_body():
#     return batch_predict(body_model, body_index_to_label)

# def batch_predict(model, index_to_label):
#     if 'images' not in request.files:
#         logger.warning("No images provided in the request")
#         return jsonify({"error": "No images provided"}), 400

#     image_files = request.files.getlist('images')

#     try:
#         total_start_time = time.time()
#         batch_images = []
#         filenames = []

#         preprocess_start_time = time.time()
#         for file in image_files:
#             try:
#                 img_bytes = file.read()
#                 img_array = preprocess_image(img_bytes)
#                 batch_images.append(img_array)
#                 filenames.append(file.filename)
#             except Exception as e:
#                 logger.error(f"Error processing image {file.filename}: {str(e)}")
#                 # Skip this image and continue with the rest

#         preprocess_end_time = time.time()
#         image_processing_time = preprocess_end_time - preprocess_start_time

#         if not batch_images:
#             return jsonify({"error": "No valid images in the batch"}), 400

#         prediction_start_time = time.time()
#         batch_predictions = predict_image(np.array(batch_images), model, index_to_label)
#         prediction_end_time = time.time()
#         prediction_time = prediction_end_time - prediction_start_time

#         total_end_time = time.time()
#         total_processing_time = total_end_time - total_start_time

#         batch_results = [
#             {
#                 "filename": filename,
#                 "predictions": predictions
#             }
#             for filename, predictions in zip(filenames, batch_predictions)
#         ]

#         logger.info(f"Batch prediction completed. Processed {len(batch_images)} images in {total_processing_time:.4f} seconds")

#         return jsonify({
#             "results": batch_results,
#             "image_processing_time": image_processing_time,
#             "prediction_time": prediction_time,
#             "total_processing_time": total_processing_time
#         })
#     except Exception as e:
#         logger.error(f"Error during batch prediction: {str(e)}", exc_info=True)
#         return jsonify({"error": "An error occurred during batch prediction"}), 500

# @model.route("/modelInfo", methods=["GET"])
# def get_model_info():
#     return jsonify({
#         "message": "Models are ready for predictions",
#         "available_models": ["face", "body"],
#         "prediction_endpoints": [
#             "/predict/face",
#             "/predict/body",
#             "/batchPredict/face",
#             "/batchPredict/body"
#         ],
#         "model_type": "MobileNetV3Small",
#         "face_model_info": {
#             "input_shape": face_model.input_shape[1:],
#             "num_classes": len(face_label_to_index),
#             "label_mapping": face_label_to_index
#         },
#         "body_model_info": {
#             "input_shape": body_model.input_shape[1:],
#             "num_classes": len(body_label_to_index),
#             "label_mapping": body_label_to_index
#         },
#         "prediction_threshold": PREDICTION_THRESHOLD,
#         "max_image_size": MAX_IMAGE_SIZE,
#         "allowed_image_types": list(ALLOWED_IMAGE_TYPES)
#     })
