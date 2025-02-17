from flask import Blueprint, Response, make_response, request
import logging
import json
import time
from queue import Queue
from firestore import obd_drive_sessions  # Use this for writing to DB

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create blueprint with new name
realtime_obd_stream_handling = Blueprint("realtime_obd_stream_handling", __name__)

# Buffer for OBD data
obd_data_buffer = Queue(maxsize=1)


# TODO: Save to the database if we are on a different second based on server time
# TODO: DB schema - obd_drive_sessions/<sessionId>/obd_drive_session_data/<timestamp>
# TODO: DB schema - there will be a field for each of the things we want - speed, rpm, etc
# Look at DB handling in realtime_camera_stream_handling.py for reference
@realtime_obd_stream_handling.route("/obd_data", methods=["POST"])
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
                    # Commented out fields can be uncommented as needed
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


@realtime_obd_stream_handling.route("/obd_stream_view")
def obd_stream_view():
    global obd_data_buffer

    def generate():
        while True:
            try:
                while not obd_data_buffer.empty():
                    data = obd_data_buffer.get_nowait()
                    yield f"data: {json.dumps(data)}\n\n"
            except Queue.Empty:
                pass
            time.sleep(0.1)

    return Response(generate(), mimetype="text/event-stream")
