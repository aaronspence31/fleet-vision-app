#include <Arduino_JSON.h>
#include <OBD2UART.h>
#include <string.h>
#include <stdlib.h>

COBD obd;

// ----- Global Light Group Mapping -----
// Define four groups with full sensor lists.
struct LightGroup {
  const char* system;
  const char* sensors[10];  // Maximum sensors per group
  byte sensorCount;
};

LightGroup groups[] = {
  { "check_engine", { 
      "Oxygen (O₂) Sensors", 
      "Mass Air Flow (MAF) Sensor", 
      "Engine Coolant Temperature (ECT) Sensor", 
      "Throttle Position Sensor (TPS)", 
      "Manifold Absolute Pressure (MAP) Sensor", 
      "Crankshaft Position Sensor", 
      "Camshaft Position Sensor", 
      "Knock Sensor", 
      "Evaporative Emission (EVAP) System Sensors" 
    }, 9 },
  { "transmission", { 
      "Transmission Fluid Temperature Sensor", 
      "Transmission Speed Sensor", 
      "Transmission Pressure Sensor" 
    }, 3 },
  { "abs", { 
      "Wheel Speed Sensors", 
      "Brake Pressure Sensors", 
      "ABS Control Module/Related Circuitry" 
    }, 3 },
  { "airbag", { 
      "Crash/Impact Sensors", 
      "Occupant Detection Sensors", 
      "Airbag Module Self-Diagnostics" 
    }, 3 }
};

const int numGroups = sizeof(groups) / sizeof(groups[0]);

// ----- Helper: Classify a DTC Code into a System -----
// Simple rules:
// - 'P' codes: if they begin with "P07" then transmission; otherwise check_engine.
// - 'C' codes: abs.
// - 'B' codes: airbag.
const char* classifyDTC(const char* dtc) {
  if (dtc[0] == 'P') {
    if (dtc[1] == '0' && dtc[2] == '7')
      return "transmission";
    else
      return "check_engine";
  } else if (dtc[0] == 'C') {
    return "abs";
  } else if (dtc[0] == 'B') {
    return "airbag";
  }
  return NULL;
}

// ----- Convert a 16-bit DTC Code to a String (e.g., "P0123") -----
// The OBD2UART readDTC() function returns a 16-bit code.
void formatDTC(uint16_t code, char* dtcStr, size_t dtcStrSize) {
  uint8_t byte1 = (code >> 8) & 0xFF;
  uint8_t byte2 = code & 0xFF;
  char type;
  uint8_t t = (byte1 & 0xC0) >> 6;
  switch (t) {
    case 0: type = 'P'; break;
    case 1: type = 'C'; break;
    case 2: type = 'B'; break;
    case 3: type = 'U'; break;
    default: type = 'X'; break;
  }
  int digit1 = (byte1 & 0x30) >> 4;
  int digit2 = (byte1 & 0x0F);
  int digit3 = (byte2 >> 4);
  int digit4 = (byte2 & 0x0F);
  snprintf(dtcStr, dtcStrSize, "%c%d%d%d%d", type, digit1, digit2, digit3, digit4);
}

// ----- Standard PID Functions -----
bool getSpeed(int &speed) {
  return obd.readPID(PID_SPEED, speed);
}

bool getRPM(int &rpm) {
  return obd.readPID(PID_RPM, rpm);
}

// ----- Get Warning Status and Build Sensor Groups -----
// Reads Mode 01 PID 0x01 to check MIL status and DTC count.
// If MIL is on, retrieves DTC codes via obd.readDTC() and classifies them.
// For each active group, adds the full sensor list from our mapping.
void getWarningStatus(JSONVar &jsonObj) {
  int value;
  if (obd.readPID(0x01, value)) {
    bool milOn = (value & 0x80) != 0;
    jsonObj["check_engine_on"] = milOn;
    jsonObj["num_dtc_codes"] = value & 0x7F;
    
    if (milOn) {
      // Retrieve up to 8 DTC codes.
      uint16_t codes[8];
      byte numCodes = obd.readDTC(codes, 8);
      
      char rawDTCs[128] = "";
      bool first = true;
      // Flags for active groups (order: check_engine, transmission, abs, airbag)
      bool activeGroup[4] = { false, false, false, false };
      
      for (byte i = 0; i < numCodes; i++) {
        char dtcStr[6];
        formatDTC(codes[i], dtcStr, sizeof(dtcStr));
        
        if (!first) {
          strncat(rawDTCs, ",", sizeof(rawDTCs) - strlen(rawDTCs) - 1);
        }
        strncat(rawDTCs, dtcStr, sizeof(rawDTCs) - strlen(rawDTCs) - 1);
        first = false;
        
        const char* system = classifyDTC(dtcStr);
        if (system) {
          for (int j = 0; j < numGroups; j++) {
            if (strcmp(groups[j].system, system) == 0) {
              activeGroup[j] = true;
            }
          }
        }
      }
      jsonObj["dtc_codes"] = String(rawDTCs);
      
      JSONVar warningLights = JSON.parse("{}");
      for (int j = 0; j < numGroups; j++) {
        if (activeGroup[j]) {
          JSONVar groupObj = JSON.parse("{}");
          JSONVar sensors = JSON.parse("[]");
          for (byte k = 0; k < groups[j].sensorCount; k++) {
            sensors[k] = groups[j].sensors[k];
          }
          groupObj["sensors"] = sensors;
          warningLights[groups[j].system] = groupObj;
        }
      }
      jsonObj["warning_lights"] = warningLights;
    }
  }
}

// ----- Build and Print JSON Data -----
void sendJsonData() {
  JSONVar jsonObj = JSON.parse("{}");
  jsonObj["timestamp"] = millis();
  
  int speed = 0, rpm = 0;
  if (getSpeed(speed))
    jsonObj["speed"] = speed;
  else
    jsonObj["speed"] = -1;
  
  if (getRPM(rpm))
    jsonObj["rpm"] = rpm;
  else
    jsonObj["rpm"] = -1;
  
  getWarningStatus(jsonObj);
  
  String jsonString = JSON.stringify(jsonObj);
  Serial.println(jsonString);
}

void setup() {
  Serial.begin(115200);
  Serial1.begin(9600);
  pinMode(12, OUTPUT);
  digitalWrite(12, HIGH);
  delay(100);
  
  Serial.println("Connecting to OBD...");
  while (!obd.begin()) {
    Serial.println("Failed to connect to OBD. Retrying...");
    delay(1000);
  }
  Serial.println("Connected to OBD!");
  
  while (!obd.init()) {
    Serial.println("Initializing OBD...");
    delay(1000);
  }
  Serial.println("OBD initialized!");
}

void loop() {
  sendJsonData();
  delay(1000);
}





// /*************************************************************************************************
//     OBD-II_PIDs COMBINED CODE: Vehicle Speed and Engine RPM

//     Query
//     send id: 0x7df
//       dta: 0x02, 0x01, PID_CODE, 0, 0, 0, 0, 0

//     Response
//     From id: 0x7E9 or 0x7EA or 0x7EB
//       dta: len, 0x41, PID_CODE, byte0, byte1(option), byte2(option), byte3(option), byte4(option)

//     https://en.wikipedia.org/wiki/OBD-II_PIDs
// ***************************************************************************************************/

// #include <SPI.h>
// #include <mcp_canbus.h>

// /* Please modify SPI_CS_PIN to adapt to your board.

//    CANBed V1        - 17
//    CANBed M0        - 3
//    CAN Bus Shield   - 9
//    CANBed 2040      - 9
//    CANBed Dual      - 9
//    OBD-2G Dev Kit   - 9
//    OBD-II GPS Kit   - 9
//    Hud Dev Kit      - 9

//    Seeed Studio CAN-Bus Breakout Board for XIAO and QT Py - D7
// */

// #define SPI_CS_PIN  9

// MCP_CAN CAN(SPI_CS_PIN);  // Set CS pin

// #define PID_ENGIN_PRM           0x0C  // Engine RPM
// #define PID_VEHICLE_SPEED       0x0D  // Vehicle Speed
// #define PID_COOLANT_TEMP        0x05  // Coolant Temperature
// #define PID_FUEL_STATUS         0x03  // Fuel System Status
// #define PID_INTAKE_AIR_TEMP     0x0F  // Temperature of air taken in by engine
// #define PID_MASS_AIR_FLOW       0x10  // Amount of air taken in my engine
// #define PID_ENGINE_LOAD         0x04  // Indicates how hard engine is working, used to determine inefficiencies

// // New PIDs
// #define PID_FUEL_RAIL_PRESSURE  0x22
// #define PID_FUEL_RAIL_GAUGE     0x23
// #define PID_FUEL_LEVEL_INPUT    0x2F
// #define PID_BAROMETRIC_PRESSURE 0x30
// #define PID_DISTANCE_DTC        0x31
// #define PID_DISTANCE_MIL        0x21
// #define PID_TRANS_FLUID_TEMP    0x49
// #define PID_TRANS_GEAR_RATIO    0x4F
// #define PID_WHEEL_SPEED         0x4A
// #define PID_SECURITY_STATUS     0xC1

// #define CAN_ID_PID              0x7DF // PID Request ID

// void set_mask_filt() {
//     // Set mask and filters to receive specific OBD-II responses
//     CAN.init_Mask(0, 0, 0x7FC);
//     CAN.init_Mask(1, 0, 0x7FC);
//     CAN.init_Filt(0, 0, 0x7E8);
//     CAN.init_Filt(1, 0, 0x7E8);
//     CAN.init_Filt(2, 0, 0x7E8);
//     CAN.init_Filt(3, 0, 0x7E8);
//     CAN.init_Filt(4, 0, 0x7E8);
//     CAN.init_Filt(5, 0, 0x7E8);
// }

// void sendPid(unsigned char __pid) {
//     unsigned char tmp[8] = {0x02, 0x01, __pid, 0, 0, 0, 0, 0};
//     CAN.sendMsgBuf(CAN_ID_PID, 0, 8, tmp);
// }

// bool getSpeed(int *speed) {
//     sendPid(PID_VEHICLE_SPEED);
//     unsigned long __timeout = millis();

//     while (millis() - __timeout < 1000) {  // 1s timeout
//         unsigned char len = 0;
//         unsigned char buf[8];

//         if (CAN_MSGAVAIL == CAN.checkReceive()) {  // Check if data is available
//             CAN.readMsgBuf(&len, buf);             // Read data

//             if (buf[1] == 0x41 && buf[2] == PID_VEHICLE_SPEED) {
//                 *speed = buf[3];  // Extract speed value
//                 return true;
//             }
//         }
//     }
//     return false;  // Timeout or no valid data
// }

// bool getRPM(int *rpm) {
//     sendPid(PID_ENGIN_PRM);
//     unsigned long __timeout = millis();

//     while (millis() - __timeout < 1000) {  // 1s timeout
//         unsigned char len = 0;
//         unsigned char buf[8];

//         if (CAN_MSGAVAIL == CAN.checkReceive()) {  // Check if data is available
//             CAN.readMsgBuf(&len, buf);             // Read data

//             if (buf[1] == 0x41 && buf[2] == PID_ENGIN_PRM) {
//                 *rpm = ((buf[3] * 256) + buf[4]) / 4;  // Convert RPM from bytes
//                 return true;
//             }
//         }
//     }
//     return false;  // Timeout or no valid data
// }

// bool getCoolantTemp(int *temp) {
//     sendPid(PID_COOLANT_TEMP);
//     unsigned long __timeout = millis();

//     while (millis() - __timeout < 1000) {  // 1s timeout
//         unsigned char len = 0;
//         unsigned char buf[8];

//         if (CAN_MSGAVAIL == CAN.checkReceive()) {  // Check if data is available
//             CAN.readMsgBuf(&len, buf);             // Read data

//             if (buf[1] == 0x41 && buf[2] == PID_COOLANT_TEMP) {
//                 *temp = buf[3] - 40;  // Extract and adjust temperature (OBD-II spec)
//                 return true;
//             }
//         }
//     }
//     return false;  // Timeout or no valid data
// }

// bool getFuelStatus(int *status) {
//     sendPid(PID_FUEL_STATUS);
//     unsigned long __timeout = millis();

//     while (millis() - __timeout < 1000) {  // 1s timeout
//         unsigned char len = 0;
//         unsigned char buf[8];

//         if (CAN_MSGAVAIL == CAN.checkReceive()) {  // Check if data is available
//             CAN.readMsgBuf(&len, buf);             // Read data

//             if (buf[1] == 0x41 && buf[2] == PID_FUEL_STATUS) {
//                 *status = buf[3];  // Extract fuel system status directly
//                 return true;
//             }
//         }
//     }
//     return false;  // Timeout or no valid data
// }

// bool getIntakeAirTemp(int *temp) {
//     sendPid(PID_INTAKE_AIR_TEMP);
//     unsigned long __timeout = millis();

//     while (millis() - __timeout < 1000) {  // 1s timeout
//         unsigned char len = 0;
//         unsigned char buf[8];

//         if (CAN_MSGAVAIL == CAN.checkReceive()) {  // Check if data is available
//             CAN.readMsgBuf(&len, buf);             // Read data

//             if (buf[1] == 0x41 && buf[2] == PID_INTAKE_AIR_TEMP) {
//                 *temp = buf[3] - 40;  // Intake Air Temperature = A - 40
//                 return true;
//             }
//         }
//     }
//     return false;  // Timeout or no valid data
// }

// bool getMassAirFlow(float *massAirFlow) {
//     sendPid(PID_MASS_AIR_FLOW);
//     unsigned long __timeout = millis();

//     while (millis() - __timeout < 1000) {  // 1s timeout
//         unsigned char len = 0;
//         unsigned char buf[8];

//         if (CAN_MSGAVAIL == CAN.checkReceive()) {  // Check if data is available
//             CAN.readMsgBuf(&len, buf);             // Read data

//             if (buf[1] == 0x41 && buf[2] == PID_MASS_AIR_FLOW) {
//                 *massAirFlow = ((256.0 * buf[3]) + buf[4]) / 100.0;  // Mass Air Flow = (256 * A + B) / 100
//                 return true;
//             }
//         }
//     }
//     return false;  // Timeout or no valid data
// }

// bool getEngineLoad(float *engineLoad) {
//     sendPid(PID_ENGINE_LOAD);
//     unsigned long __timeout = millis();

//     while (millis() - __timeout < 1000) {  // 1s timeout
//         unsigned char len = 0;
//         unsigned char buf[8];

//         if (CAN_MSGAVAIL == CAN.checkReceive()) {  // Check if data is available
//             CAN.readMsgBuf(&len, buf);             // Read data

//             if (buf[1] == 0x41 && buf[2] == PID_ENGINE_LOAD) {
//                 *engineLoad = ((float)buf[3] * 100.0) / 255.0;  // Engine Load (%) = (A * 100) / 255
//                 return true;
//             }
//         }
//     }
//     return false;  // Timeout or no valid data
// }

// // NEWLY ADDED PID METHODS BEGINS

// bool getFuelRailPressure(int *pressure) {
//     sendPid(PID_FUEL_RAIL_PRESSURE);
//     unsigned long __timeout = millis();
//     while (millis() - __timeout < 1000) {
//         unsigned char len = 0;
//         unsigned char buf[8];
//         if (CAN_MSGAVAIL == CAN.checkReceive()) {
//             CAN.readMsgBuf(&len, buf);
//             if (buf[1] == 0x41 && buf[2] == PID_FUEL_RAIL_PRESSURE) {
//                 *pressure = ((buf[3] * 256) + buf[4]) * 10;  // Convert to kPa
//                 return true;
//             }
//         }
//     }
//     return false;
// }

// bool getFuelRailGauge(int *gauge) {
//     sendPid(PID_FUEL_RAIL_GAUGE);
//     unsigned long __timeout = millis();
//     while (millis() - __timeout < 1000) {
//         unsigned char len = 0;
//         unsigned char buf[8];
//         if (CAN_MSGAVAIL == CAN.checkReceive()) {
//             CAN.readMsgBuf(&len, buf);
//             if (buf[1] == 0x41 && buf[2] == PID_FUEL_RAIL_GAUGE) {
//                 *gauge = ((buf[3] * 256) + buf[4]) * 10;  // Convert to kPa
//                 return true;
//             }
//         }
//     }
//     return false;
// }

// bool getFuelLevel(int *level) {
//     sendPid(PID_FUEL_LEVEL_INPUT);
//     unsigned long __timeout = millis();
//     while (millis() - __timeout < 1000) {
//         unsigned char len = 0;
//         unsigned char buf[8];
//         if (CAN_MSGAVAIL == CAN.checkReceive()) {
//             CAN.readMsgBuf(&len, buf);
//             if (buf[1] == 0x41 && buf[2] == PID_FUEL_LEVEL_INPUT) {
//                 *level = (buf[3] * 100) / 255;  // Convert to percentage
//                 return true;
//             }
//         }
//     }
//     return false;
// }

// bool getBarometricPressure(int *pressure) {
//     sendPid(PID_BAROMETRIC_PRESSURE);
//     unsigned long __timeout = millis();
//     while (millis() - __timeout < 1000) {
//         unsigned char len = 0;
//         unsigned char buf[8];
//         if (CAN_MSGAVAIL == CAN.checkReceive()) {
//             CAN.readMsgBuf(&len, buf);
//             if (buf[1] == 0x41 && buf[2] == PID_BAROMETRIC_PRESSURE) {
//                 *pressure = buf[3];  // Convert to kPa
//                 return true;
//             }
//         }
//     }
//     return false;
// }

// bool getDistanceDTC(int *distance) {
//     sendPid(PID_DISTANCE_DTC);
//     unsigned long __timeout = millis();
//     while (millis() - __timeout < 1000) {
//         unsigned char len = 0;
//         unsigned char buf[8];
//         if (CAN_MSGAVAIL == CAN.checkReceive()) {
//             CAN.readMsgBuf(&len, buf);
//             if (buf[1] == 0x41 && buf[2] == PID_DISTANCE_DTC) {
//                 *distance = (buf[3] * 256) + buf[4];  // Convert to kilometers
//                 return true;
//             }
//         }
//     }
//     return false;
// }

// bool getDistanceMIL(int *distance) {
//     sendPid(PID_DISTANCE_MIL);
//     unsigned long __timeout = millis();
//     while (millis() - __timeout < 1000) {
//         unsigned char len = 0;
//         unsigned char buf[8];
//         if (CAN_MSGAVAIL == CAN.checkReceive()) {
//             CAN.readMsgBuf(&len, buf);
//             if (buf[1] == 0x41 && buf[2] == PID_DISTANCE_MIL) {
//                 *distance = (buf[3] * 256) + buf[4];  // Convert to kilometers
//                 return true;
//             }
//         }
//     }
//     return false;
// }

// bool getTransmissionFluidTemp(int *temp) {
//     sendPid(PID_TRANS_FLUID_TEMP);
//     unsigned long __timeout = millis();
//     while (millis() - __timeout < 1000) {
//         unsigned char len = 0;
//         unsigned char buf[8];
//         if (CAN_MSGAVAIL == CAN.checkReceive()) {
//             CAN.readMsgBuf(&len, buf);
//             if (buf[1] == 0x41 && buf[2] == PID_TRANS_FLUID_TEMP) {
//                 *temp = buf[3] - 40;  // Convert to Celsius
//                 return true;
//             }
//         }
//     }
//     return false;
// }

// bool getTransmissionGearRatio(float *ratio) {
//     sendPid(PID_TRANS_GEAR_RATIO);
//     unsigned long __timeout = millis();
//     while (millis() - __timeout < 1000) {
//         unsigned char len = 0;
//         unsigned char buf[8];
//         if (CAN_MSGAVAIL == CAN.checkReceive()) {
//             CAN.readMsgBuf(&len, buf);
//             if (buf[1] == 0x41 && buf[2] == PID_TRANS_GEAR_RATIO) {
//                 *ratio = ((buf[3] * 256.0) + buf[4]) / 1000.0;  // Convert to a float
//                 return true;
//             }
//         }
//     }
//     return false;
// }

// bool getWheelSpeed(int *speed) {
//     sendPid(PID_WHEEL_SPEED);
//     unsigned long __timeout = millis();
//     while (millis() - __timeout < 1000) {
//         unsigned char len = 0;
//         unsigned char buf[8];
//         if (CAN_MSGAVAIL == CAN.checkReceive()) {
//             CAN.readMsgBuf(&len, buf);
//             if (buf[1] == 0x41 && buf[2] == PID_WHEEL_SPEED) {
//                 *speed = buf[3];  // Speed in km/h
//                 return true;
//             }
//         }
//     }
//     return false;
// }

// bool getVehicleSecurityStatus(int *status) {
//     sendPid(PID_SECURITY_STATUS);
//     unsigned long __timeout = millis();
//     while (millis() - __timeout < 1000) {
//         unsigned char len = 0;
//         unsigned char buf[8];
//         if (CAN_MSGAVAIL == CAN.checkReceive()) {
//             CAN.readMsgBuf(&len, buf);
//             if (buf[1] == 0x41 && buf[2] == PID_SECURITY_STATUS) {
//                 *status = buf[3];  // Status as a raw byte
//                 return true;
//             }
//         }
//     }
//     return false;
// }

// // NEWLY ADDED PID METHODS ENDS

// void setup() {
//     Serial.begin(115200);
//     while (!Serial);

//     // For OBD-II GPS Dev Kit RP2040 version
//     pinMode(12, OUTPUT);
//     digitalWrite(12, HIGH);

//     while (CAN_OK != CAN.begin(CAN_500KBPS)) {  // Initialize CAN bus at 500 kbps
//         Serial.println("CAN init fail, retry...");
//         delay(100);
//     }
//     Serial.println("CAN init ok!");
//     set_mask_filt();
// }

// void loop() {
//     int speed = 0;
//     int rpm = 0;
//     int coolantTemp = 0;
//     int fuelStatus = 0;
//     int intakeAirTemp = 0;
//     float massAirFlow = 0.0;
//     float engineLoad = 0.0;

//     int fuelRailPressure = 0;
//     int fuelRailGauge = 0;
//     int fuelLevel = 0;
//     int barometricPressure = 0;
//     int distanceDTC = 0;
//     int distanceMIL = 0;
//     int transFluidTemp = 0;
//     float transGearRatio = 0.0;
//     int wheelSpeed = 0;

//     Serial.println();
//     Serial.println();

//     // Get Vehicle Speed
//     if (getSpeed(&speed)) {
//         Serial.print("Vehicle Speed: ");
//         Serial.print(speed);
//         Serial.println(" km/h");
//     } else {
//         Serial.println("Failed to get vehicle speed.");
//     }

//     // Get Engine RPM
//     if (getRPM(&rpm)) {
//         Serial.print("Engine RPM: ");
//         Serial.print(rpm);
//         Serial.println(" rpm");
//     } else {
//         Serial.println("Failed to get engine RPM.");
//     }

//     // Get Coolant Temperature
//     if (getCoolantTemp(&coolantTemp)) {
//         Serial.print("Coolant Temperature: ");
//         Serial.print(coolantTemp);
//         Serial.println(" °C");
//     } else {
//         Serial.println("Failed to get coolant temperature.");
//     }

//     // Get Fuel System Status
//     if (getFuelStatus(&fuelStatus)) {
//         Serial.print("Fuel System Status: ");
//         Serial.println(fuelStatus);
//     } else {
//         Serial.println("Failed to get fuel system status.");
//     }

//     // Get Intake Air Temperature
//     if (getIntakeAirTemp(&intakeAirTemp)) {
//         Serial.print("Intake Air Temperature: ");
//         Serial.print(intakeAirTemp);
//         Serial.println(" °C");
//     } else {
//         Serial.println("Failed to get intake air temperature.");
//     }

//     // Get Mass Air Flow
//     if (getMassAirFlow(&massAirFlow)) {
//         Serial.print("Mass Air Flow: ");
//         Serial.print(massAirFlow, 2);  // Two decimal places
//         Serial.println(" g/s");
//     } else {
//         Serial.println("Failed to get mass air flow.");
//     }

//     // Get Engine Load
//     if (getEngineLoad(&engineLoad)) {
//         Serial.print("Engine Load: ");
//         Serial.print(engineLoad, 1);  // One decimal place
//         Serial.println(" %");
//     } else {
//         Serial.println("Failed to get engine load.");
//     }

//     // Get Fuel Rail Pressure
//     if (getFuelRailPressure(&fuelRailPressure)) {
//         Serial.print("Fuel Rail Pressure: ");
//         Serial.print(fuelRailPressure);
//         Serial.println(" kPa");
//     } else {
//         Serial.println("Failed to get fuel rail pressure.");
//     }

//     // Get Fuel Rail Gauge Pressure
//     if (getFuelRailGauge(&fuelRailGauge)) {
//         Serial.print("Fuel Rail Gauge Pressure: ");
//         Serial.print(fuelRailGauge);
//         Serial.println(" kPa");
//     } else {
//         Serial.println("Failed to get fuel rail gauge pressure.");
//     }

//     // Get Fuel Level
//     if (getFuelLevel(&fuelLevel)) {
//         Serial.print("Fuel Level: ");
//         Serial.print(fuelLevel);
//         Serial.println(" %");
//     } else {
//         Serial.println("Failed to get fuel level.");
//     }

//     // Get Barometric Pressure
//     if (getBarometricPressure(&barometricPressure)) {
//         Serial.print("Barometric Pressure: ");
//         Serial.print(barometricPressure);
//         Serial.println(" kPa");
//     } else {
//         Serial.println("Failed to get barometric pressure.");
//     }

//     // Get Distance Since DTC Cleared
//     if (getDistanceDTC(&distanceDTC)) {
//         Serial.print("Distance Since DTC Cleared: ");
//         Serial.print(distanceDTC);
//         Serial.println(" km");
//     } else {
//         Serial.println("Failed to get distance since DTC cleared.");
//     }

//     // Get Distance Traveled with MIL On
//     if (getDistanceMIL(&distanceMIL)) {
//         Serial.print("Distance with MIL On: ");
//         Serial.print(distanceMIL);
//         Serial.println(" km");
//     } else {
//         Serial.println("Failed to get distance traveled with MIL on.");
//     }

//     // Get Transmission Fluid Temperature
//     if (getTransmissionFluidTemp(&transFluidTemp)) {
//         Serial.print("Transmission Fluid Temperature: ");
//         Serial.print(transFluidTemp);
//         Serial.println(" °C");
//     } else {
//         Serial.println("Failed to get transmission fluid temperature.");
//     }

//     // Get Transmission Gear Ratio
//     if (getTransmissionGearRatio(&transGearRatio)) {
//         Serial.print("Transmission Gear Ratio: ");
//         Serial.print(transGearRatio, 3);  // Three decimal places
//         Serial.println();
//     } else {
//         Serial.println("Failed to get transmission gear ratio.");
//     }

//     // Get Wheel Speed
//     if (getWheelSpeed(&wheelSpeed)) {
//         Serial.print("Wheel Speed: ");
//         Serial.print(wheelSpeed);
//         Serial.println(" km/h");
//     } else {
//         Serial.println("Failed to get wheel speed.");
//     }

//     Serial.println();
//     Serial.println();

//     delay(5000);  // Query every 5 seconds
// }

// // END FILE
