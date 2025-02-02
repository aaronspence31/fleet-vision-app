from flask import Flask
from routes.stream_viewer import stream_viewer
import logging
from flask_cors import CORS

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.register_blueprint(stream_viewer)
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
    logger.info("Starting Flask app...")
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)
    logger.info("Server stopped.")  # This line will execute when the server stops
