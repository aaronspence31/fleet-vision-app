#include <SPI.h>
#include <mcp_canbus.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <HTTPClient.h>

// Wi-Fi credentials
const char* ssid = "Isaiahs Phone";
const char* password = "12345678";

// Flask server URL
const char* serverUrl = "http://127.0.0.1:5000/body_stream_clip_view"; 

#define SPI_CS_PIN  9
MCP_CAN CAN(SPI_CS_PIN);  // Set CS pin

// OBD-II PIDs
#define PID_ENGIN_PRM           0x0C  // Engine RPM
#define PID_VEHICLE_SPEED       0x0D  // Vehicle Speed
#define PID_COOLANT_TEMP        0x05  // Coolant Temperature
#define PID_FUEL_STATUS         0x03  // Fuel System Status
#define PID_INTAKE_AIR_TEMP     0x0F  // Intake Air Temperature
#define PID_MASS_AIR_FLOW       0x10  // Mass Air Flow
#define PID_ENGINE_LOAD         0x04  // Engine Load

// New PIDs
#define PID_FUEL_RAIL_PRESSURE  0x22
#define PID_FUEL_RAIL_GAUGE     0x23
#define PID_FUEL_LEVEL_INPUT    0x2F
#define PID_BAROMETRIC_PRESSURE 0x30
#define PID_DISTANCE_DTC        0x31
#define PID_DISTANCE_MIL        0x21
#define PID_TRANS_FLUID_TEMP    0x49
#define PID_TRANS_GEAR_RATIO    0x4F
#define PID_WHEEL_SPEED         0x4A

#define CAN_ID_PID              0x7DF // PID Request ID

// Set CAN mask and filters to only receive relevant responses
void set_mask_filt() {
  CAN.init_Mask(0, 0, 0x7FC);
  CAN.init_Mask(1, 0, 0x7FC);
  CAN.init_Filt(0, 0, 0x7E8);
  CAN.init_Filt(1, 0, 0x7E8);
  CAN.init_Filt(2, 0, 0x7E8);
  CAN.init_Filt(3, 0, 0x7E8);
  CAN.init_Filt(4, 0, 0x7E8);
  CAN.init_Filt(5, 0, 0x7E8);
}

// Sends a PID request message over the CAN bus
void sendPid(unsigned char __pid) {
  unsigned char tmp[8] = {0x02, 0x01, __pid, 0, 0, 0, 0, 0};
  CAN.sendMsgBuf(CAN_ID_PID, 0, 8, tmp);
}

// Standard OBD-II functions:

bool getSpeed(int *speed) {
  sendPid(PID_VEHICLE_SPEED);
  unsigned long __timeout = millis();
  while (millis() - __timeout < 1000) {
    unsigned char len = 0;
    unsigned char buf[8];
    if (CAN_MSGAVAIL == CAN.checkReceive()) {
      CAN.readMsgBuf(&len, buf);
      if (buf[1] == 0x41 && buf[2] == PID_VEHICLE_SPEED) {
        *speed = buf[3];
        return true;
      }
    }
  }
  return false;
}

bool getRPM(int *rpm) {
  sendPid(PID_ENGIN_PRM);
  unsigned long __timeout = millis();
  while (millis() - __timeout < 1000) {
    unsigned char len = 0;
    unsigned char buf[8];
    if (CAN_MSGAVAIL == CAN.checkReceive()) {
      CAN.readMsgBuf(&len, buf);
      if (buf[1] == 0x41 && buf[2] == PID_ENGIN_PRM) {
        *rpm = ((buf[3] * 256) + buf[4]) / 4;
        return true;
      }
    }
  }
  return false;
}

bool getCoolantTemp(int *temp) {
  sendPid(PID_COOLANT_TEMP);
  unsigned long __timeout = millis();
  while (millis() - __timeout < 1000) {
    unsigned char len = 0;
    unsigned char buf[8];
    if (CAN_MSGAVAIL == CAN.checkReceive()) {
      CAN.readMsgBuf(&len, buf);
      if (buf[1] == 0x41 && buf[2] == PID_COOLANT_TEMP) {
        *temp = buf[3] - 40;
        return true;
      }
    }
  }
  return false;
}

bool getFuelStatus(int *status) {
  sendPid(PID_FUEL_STATUS);
  unsigned long __timeout = millis();
  while (millis() - __timeout < 1000) {
    unsigned char len = 0;
    unsigned char buf[8];
    if (CAN_MSGAVAIL == CAN.checkReceive()) {
      CAN.readMsgBuf(&len, buf);
      if (buf[1] == 0x41 && buf[2] == PID_FUEL_STATUS) {
        *status = buf[3];
        return true;
      }
    }
  }
  return false;
}

bool getIntakeAirTemp(int *temp) {
  sendPid(PID_INTAKE_AIR_TEMP);
  unsigned long __timeout = millis();
  while (millis() - __timeout < 1000) {
    unsigned char len = 0;
    unsigned char buf[8];
    if (CAN_MSGAVAIL == CAN.checkReceive()) {
      CAN.readMsgBuf(&len, buf);
      if (buf[1] == 0x41 && buf[2] == PID_INTAKE_AIR_TEMP) {
        *temp = buf[3] - 40;
        return true;
      }
    }
  }
  return false;
}

bool getMassAirFlow(float *massAirFlow) {
  sendPid(PID_MASS_AIR_FLOW);
  unsigned long __timeout = millis();
  while (millis() - __timeout < 1000) {
    unsigned char len = 0;
    unsigned char buf[8];
    if (CAN_MSGAVAIL == CAN.checkReceive()) {
      CAN.readMsgBuf(&len, buf);
      if (buf[1] == 0x41 && buf[2] == PID_MASS_AIR_FLOW) {
        *massAirFlow = ((256.0 * buf[3]) + buf[4]) / 100.0;
        return true;
      }
    }
  }
  return false;
}

bool getEngineLoad(float *engineLoad) {
  sendPid(PID_ENGINE_LOAD);
  unsigned long __timeout = millis();
  while (millis() - __timeout < 1000) {
    unsigned char len = 0;
    unsigned char buf[8];
    if (CAN_MSGAVAIL == CAN.checkReceive()) {
      CAN.readMsgBuf(&len, buf);
      if (buf[1] == 0x41 && buf[2] == PID_ENGINE_LOAD) {
        *engineLoad = (buf[3] * 100.0) / 255.0;
        return true;
      }
    }
  }
  return false;
}

bool getFuelRailPressure(int *pressure) {
  sendPid(PID_FUEL_RAIL_PRESSURE);
  unsigned long __timeout = millis();
  while (millis() - __timeout < 1000) {
    unsigned char len = 0;
    unsigned char buf[8];
    if (CAN_MSGAVAIL == CAN.checkReceive()) {
      CAN.readMsgBuf(&len, buf);
      if (buf[1] == 0x41 && buf[2] == PID_FUEL_RAIL_PRESSURE) {
        *pressure = ((buf[3] * 256) + buf[4]) * 10;
        return true;
      }
    }
  }
  return false;
}

bool getFuelRailGauge(int *gauge) {
  sendPid(PID_FUEL_RAIL_GAUGE);
  unsigned long __timeout = millis();
  while (millis() - __timeout < 1000) {
    unsigned char len = 0;
    unsigned char buf[8];
    if (CAN_MSGAVAIL == CAN.checkReceive()) {
      CAN.readMsgBuf(&len, buf);
      if (buf[1] == 0x41 && buf[2] == PID_FUEL_RAIL_GAUGE) {
        *gauge = ((buf[3] * 256) + buf[4]) * 10;
        return true;
      }
    }
  }
  return false;
}

bool getFuelLevel(int *level) {
  sendPid(PID_FUEL_LEVEL_INPUT);
  unsigned long __timeout = millis();
  while (millis() - __timeout < 1000) {
    unsigned char len = 0;
    unsigned char buf[8];
    if (CAN_MSGAVAIL == CAN.checkReceive()) {
      CAN.readMsgBuf(&len, buf);
      if (buf[1] == 0x41 && buf[2] == PID_FUEL_LEVEL_INPUT) {
        *level = (buf[3] * 100) / 255;
        return true;
      }
    }
  }
  return false;
}

bool getBarometricPressure(int *pressure) {
  sendPid(PID_BAROMETRIC_PRESSURE);
  unsigned long __timeout = millis();
  while (millis() - __timeout < 1000) {
    unsigned char len = 0;
    unsigned char buf[8];
    if (CAN_MSGAVAIL == CAN.checkReceive()) {
      CAN.readMsgBuf(&len, buf);
      if (buf[1] == 0x41 && buf[2] == PID_BAROMETRIC_PRESSURE) {
        *pressure = buf[3];
        return true;
      }
    }
  }
  return false;
}

bool getDistanceDTC(int *distance) {
  sendPid(PID_DISTANCE_DTC);
  unsigned long __timeout = millis();
  while (millis() - __timeout < 1000) {
    unsigned char len = 0;
    unsigned char buf[8];
    if (CAN_MSGAVAIL == CAN.checkReceive()) {
      CAN.readMsgBuf(&len, buf);
      if (buf[1] == 0x41 && buf[2] == PID_DISTANCE_DTC) {
        *distance = (buf[3] * 256) + buf[4];
        return true;
      }
    }
  }
  return false;
}

bool getDistanceMIL(int *distance) {
  sendPid(PID_DISTANCE_MIL);
  unsigned long __timeout = millis();
  while (millis() - __timeout < 1000) {
    unsigned char len = 0;
    unsigned char buf[8];
    if (CAN_MSGAVAIL == CAN.checkReceive()) {
      CAN.readMsgBuf(&len, buf);
      if (buf[1] == 0x41 && buf[2] == PID_DISTANCE_MIL) {
        *distance = (buf[3] * 256) + buf[4];
        return true;
      }
    }
  }
  return false;
}

bool getTransmissionFluidTemp(int *temp) {
  sendPid(PID_TRANS_FLUID_TEMP);
  unsigned long __timeout = millis();
  while (millis() - __timeout < 1000) {
    unsigned char len = 0;
    unsigned char buf[8];
    if (CAN_MSGAVAIL == CAN.checkReceive()) {
      CAN.readMsgBuf(&len, buf);
      if (buf[1] == 0x41 && buf[2] == PID_TRANS_FLUID_TEMP) {
        *temp = buf[3] - 40;
        return true;
      }
    }
  }
  return false;
}

bool getTransmissionGearRatio(float *ratio) {
  sendPid(PID_TRANS_GEAR_RATIO);
  unsigned long __timeout = millis();
  while (millis() - __timeout < 1000) {
    unsigned char len = 0;
    unsigned char buf[8];
    if (CAN_MSGAVAIL == CAN.checkReceive()) {
      CAN.readMsgBuf(&len, buf);
      if (buf[1] == 0x41 && buf[2] == PID_TRANS_GEAR_RATIO) {
        *ratio = ((buf[3] * 256.0) + buf[4]) / 1000.0;
        return true;
      }
    }
  }
  return false;
}

bool getWheelSpeed(int *speed) {
  sendPid(PID_WHEEL_SPEED);
  unsigned long __timeout = millis();
  while (millis() - __timeout < 1000) {
    unsigned char len = 0;
    unsigned char buf[8];
    if (CAN_MSGAVAIL == CAN.checkReceive()) {
      CAN.readMsgBuf(&len, buf);
      if (buf[1] == 0x41 && buf[2] == PID_WHEEL_SPEED) {
        *speed = buf[3];
        return true;
      }
    }
  }
  return false;
}

// Build and send JSON data to the server
void sendDataToServer(
  int speed, int rpm, int coolantTemp,
  int fuelStatus, int intakeAirTemp, float massAirFlow, float engineLoad,
  int fuelRailPressure, int fuelRailGauge, int fuelLevel,
  int barometricPressure, int distanceDTC, int distanceMIL,
  int transFluidTemp, float transGearRatio, int wheelSpeed) {

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);

    String jsonData = "{";
    jsonData += "\"vehicle_speed\":" + String(speed) + ",";
    jsonData += "\"engine_rpm\":" + String(rpm) + ",";
    jsonData += "\"coolant_temp\":" + String(coolantTemp) + ",";
    jsonData += "\"fuel_status\":" + String(fuelStatus) + ",";
    jsonData += "\"intake_air_temp\":" + String(intakeAirTemp) + ",";
    jsonData += "\"mass_air_flow\":" + String(massAirFlow, 2) + ",";
    jsonData += "\"engine_load\":" + String(engineLoad, 1) + ",";
    jsonData += "\"fuel_rail_pressure\":" + String(fuelRailPressure) + ",";
    jsonData += "\"fuel_rail_gauge\":" + String(fuelRailGauge) + ",";
    jsonData += "\"fuel_level\":" + String(fuelLevel) + ",";
    jsonData += "\"barometric_pressure\":" + String(barometricPressure) + ",";
    jsonData += "\"distance_since_dtc_cleared\":" + String(distanceDTC) + ",";
    jsonData += "\"distance_mil_on\":" + String(distanceMIL) + ",";
    jsonData += "\"transmission_fluid_temp\":" + String(transFluidTemp) + ",";
    jsonData += "\"transmission_gear_ratio\":" + String(transGearRatio, 3) + ",";
    jsonData += "\"wheel_speed\":" + String(wheelSpeed);
    jsonData += "}";

    http.addHeader("Content-Type", "application/json");

    int httpResponseCode = http.POST(jsonData);

    if (httpResponseCode > 0) {
      Serial.print("Server response: ");
      Serial.println(http.getString());
    } else {
      Serial.print("Error sending data: ");
      Serial.println(httpResponseCode);
    }
    http.end();
  } else {
    Serial.println("WiFi not connected!");
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  pinMode(12, OUTPUT);
  digitalWrite(12, HIGH);

  // Initialize CAN bus at 500 kbps
  while (CAN_OK != CAN.begin(CAN_500KBPS)) {
    Serial.println("CAN init fail, retry...");
    delay(100);
  }
  Serial.println("CAN init ok!");
  set_mask_filt();

  // Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi!");
}

void loop() {
  int speed = 0, rpm = 0, coolantTemp = 0;
  int fuelStatus = 0, intakeAirTemp = 0;
  float massAirFlow = 0.0, engineLoad = 0.0;
  int fuelRailPressure = 0, fuelRailGauge = 0, fuelLevel = 0;
  int barometricPressure = 0, distanceDTC = 0, distanceMIL = 0;
  int transFluidTemp = 0;
  float transGearRatio = 0.0;
  int wheelSpeed = 0;

  // Retrieve data from the vehicle:
  if (getSpeed(&speed))
    Serial.print("Vehicle Speed: " + String(speed));
  else
    Serial.println("Failed to get vehicle speed.");

  if (getRPM(&rpm))
    Serial.print("Engine RPM: " + String(rpm));
  else
    Serial.println("Failed to get engine RPM.");

  if (getCoolantTemp(&coolantTemp))
    Serial.print("Coolant Temperature: " + String(coolantTemp));
  else
    Serial.println("Failed to get coolant temperature.");

  if (getFuelStatus(&fuelStatus))
    Serial.print("Fuel System Status: " + String(fuelStatus));
  else
    Serial.println("Failed to get fuel system status.");

  if (getIntakeAirTemp(&intakeAirTemp))
    Serial.print("Intake Air Temperature: " + String(intakeAirTemp));
  else
    Serial.println("Failed to get intake air temperature.");

  if (getMassAirFlow(&massAirFlow))
    Serial.print("Mass Air Flow: " + String(massAirFlow));
  else
    Serial.println("Failed to get mass air flow.");

  if (getEngineLoad(&engineLoad))
    Serial.print("Engine Load: " + String(engineLoad));
  else
    Serial.println("Failed to get engine load.");

  if (getFuelRailPressure(&fuelRailPressure))
    Serial.print("Fuel Rail Pressure: " + String(fuelRailPressure));
  else
    Serial.println("Failed to get fuel rail pressure.");

  if (getFuelRailGauge(&fuelRailGauge))
    Serial.print("Fuel Rail Gauge: " + String(fuelRailGauge));
  else
    Serial.println("Failed to get fuel rail gauge.");

  if (getFuelLevel(&fuelLevel))
    Serial.print("Fuel Level: " + String(fuelLevel));
  else
    Serial.println("Failed to get fuel level.");

  if (getBarometricPressure(&barometricPressure))
    Serial.print("Barometric Pressure: " + String(barometricPressure));
  else
    Serial.println("Failed to get barometric pressure.");

  if (getDistanceDTC(&distanceDTC))
    Serial.print("Distance Since DTC Cleared: " + String(distanceDTC));
  else
    Serial.println("Failed to get distance since DTC cleared.");

  if (getDistanceMIL(&distanceMIL))
    Serial.print("Distance MIL On: " + String(distanceMIL));
  else
    Serial.println("Failed to get distance with MIL on.");

  if (getTransmissionFluidTemp(&transFluidTemp))
    Serial.print("Transmission Fluid Temperature: " + String(transFluidTemp));
  else
    Serial.println("Failed to get transmission fluid temperature.");

  if (getTransmissionGearRatio(&transGearRatio))
    Serial.print("Transmission Gear Ratio: " + String(transGearRatio));
  else
    Serial.println("Failed to get transmission gear ratio.");

  if (getWheelSpeed(&wheelSpeed))
    Serial.print("Wheel Speed: " + String(wheelSpeed));
  else
    Serial.println("Failed to get wheel speed.");

  // Send all collected data as JSON to the server
  sendDataToServer(
    speed, rpm, coolantTemp,
    fuelStatus, intakeAirTemp, massAirFlow, engineLoad,
    fuelRailPressure, fuelRailGauge, fuelLevel,
    barometricPressure, distanceDTC, distanceMIL,
    transFluidTemp, transGearRatio, wheelSpeed
  );

  delay(5000);  // Wait 5 seconds before the next query cycle
}
