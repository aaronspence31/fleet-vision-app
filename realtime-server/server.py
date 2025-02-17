from flask import Flask
from realtime_camera_stream_handling import realtime_camera_stream_handling
from realtime_obd_stream_handling import realtime_obd_stream_handling
import logging
from flask_cors import CORS

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.register_blueprint(realtime_camera_stream_handling)
app.register_blueprint(realtime_obd_stream_handling)
CORS(app)


@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    # Return a custom error page
    return "An unexpected error occurred. Please check the server logs.", 500


if __name__ == "__main__":
    logger.info("Initializing server...")

    # Run the Flask app
    logger.info("Starting Fleet Vision real-time server...")
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False, threaded=True)
    logger.info("Server stopped.")
