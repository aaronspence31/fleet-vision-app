from flask import Flask
from realtime_stream_handling import realtime_camera_stream_handling, realtime_obd_stream_handling
import logging
from flask_cors import CORS
import os
from pyngrok import ngrok
from dotenv import load_dotenv

NGROK_HOSTANME = "ghastly-singular-snake.ngrok.app"

# Load environment variables from .env file
load_dotenv()

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

    # -------------------------------
    # Pyngrok Tunnel Setup
    # -------------------------------
    # Retrieve the ngrok auth token from environment variables loaded from .env.
    ngrok_auth_token = os.environ.get("NGROK_AUTH_TOKEN", "")
    if not ngrok_auth_token:
        raise RuntimeError("NGROK_AUTH_TOKEN environment variable not set!")
    ngrok.set_auth_token(ngrok_auth_token)

    # Define the local port your Flask app will run on.
    port = 5000

    # Open the ngrok tunnel using your reserved stable hostname.
    # Replace "ghastly-singular-snake.ngrok.app" with your actual reserved hostname.
    tunnel = ngrok.connect(addr=port, hostname=NGROK_HOSTANME)
    logger.info(f"Ngrok tunnel opened at: {tunnel.public_url}")

    # -------------------------------
    # Start Flask Server
    # -------------------------------
    logger.info("Starting Fleet Vision real-time server...")
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False, threaded=True)
    logger.info("Server stopped.")
