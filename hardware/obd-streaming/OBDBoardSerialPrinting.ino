#include <SPI.h>
#include <mcp_canbus.h>
#include <Arduino_JSON.h>
#include <string.h>
#include <stdlib.h>

// --- CAN Bus Setup ---
#define SPI_CS_PIN 9
MCP_CAN CAN(SPI_CS_PIN);

// --- OBD-II PID Definitions ---
#define PID_RPM 0x0C
#define PID_SPEED 0x0D
// Mode 01 PID 0x01 returns MIL status & DTC count

// --- Global Light Group Mapping ---
// Four groups with full sensor lists.
struct LightGroup
{
  const char *system;
  const char *sensors[10]; // Maximum sensors per group
  byte sensorCount;
};

LightGroup groups[] = {
    {"check_engine", {"Oxygen (O₂) Sensors", "Mass Air Flow (MAF) Sensor", "Engine Coolant Temperature (ECT) Sensor", "Throttle Position Sensor (TPS)", "Manifold Absolute Pressure (MAP) Sensor", "Crankshaft Position Sensor", "Camshaft Position Sensor", "Knock Sensor", "Evaporative Emission (EVAP) System Sensors"}, 9},
    {"transmission", {"Transmission Fluid Temperature Sensor", "Transmission Speed Sensor", "Transmission Pressure Sensor"}, 3},
    {"abs", {"Wheel Speed Sensors", "Brake Pressure Sensors", "ABS Control Module/Related Circuitry"}, 3},
    {"airbag", {"Crash/Impact Sensors", "Occupant Detection Sensors", "Airbag Module Self-Diagnostics"}, 3}};

const int numGroups = sizeof(groups) / sizeof(groups[0]);

// --- Helper: Classify a DTC Code into a System ---
// Simple rules:
// - 'P' codes: if they begin with "P07" then transmission; otherwise check_engine.
// - 'C' codes: abs.
// - 'B' codes: airbag.
const char *classifyDTC(const char *dtc)
{
  if (dtc[0] == 'P')
  {
    if (dtc[1] == '0' && dtc[2] == '7')
      return "transmission";
    else
      return "check_engine";
  }
  else if (dtc[0] == 'C')
  {
    return "abs";
  }
  else if (dtc[0] == 'B')
  {
    return "airbag";
  }
  return NULL;
}

// --- Helper: Convert a 16-bit DTC Code to a String (e.g., "P0123") ---
// The readDTC() response returns a 16-bit code.
void formatDTC(uint16_t code, char *dtcStr, size_t dtcStrSize)
{
  uint8_t byte1 = (code >> 8) & 0xFF;
  uint8_t byte2 = code & 0xFF;
  char type;
  uint8_t t = (byte1 & 0xC0) >> 6;
  switch (t)
  {
  case 0:
    type = 'P';
    break;
  case 1:
    type = 'C';
    break;
  case 2:
    type = 'B';
    break;
  case 3:
    type = 'U';
    break;
  default:
    type = 'X';
    break;
  }
  int digit1 = (byte1 & 0x30) >> 4;
  int digit2 = (byte1 & 0x0F);
  int digit3 = (byte2 >> 4);
  int digit4 = (byte2 & 0x0F);
  snprintf(dtcStr, dtcStrSize, "%c%d%d%d%d", type, digit1, digit2, digit3, digit4);
}

// --- Function: Send a PID Request ---
// Sends an 8-byte message with Mode 01 and the given PID.
void sendPid(byte pid)
{
  byte data[8] = {0x02, 0x01, pid, 0, 0, 0, 0, 0};
  CAN.sendMsgBuf(0x7DF, 0, 8, data);
}

// --- Standard PID Functions for Speed and RPM ---
bool getSpeed(int *speed)
{
  sendPid(PID_SPEED);
  unsigned long timeout = millis();
  while (millis() - timeout < 1000)
  {
    if (CAN_MSGAVAIL == CAN.checkReceive())
    {
      byte len = 0;
      byte buf[8];
      CAN.readMsgBuf(&len, buf);
      if (buf[1] == 0x41 && buf[2] == PID_SPEED)
      {
        *speed = buf[3];
        return true;
      }
    }
  }
  return false;
}

bool getRPM(int *rpm)
{
  sendPid(PID_RPM);
  unsigned long timeout = millis();
  while (millis() - timeout < 1000)
  {
    if (CAN_MSGAVAIL == CAN.checkReceive())
    {
      byte len = 0;
      byte buf[8];
      CAN.readMsgBuf(&len, buf);
      if (buf[1] == 0x41 && buf[2] == PID_RPM)
      {
        *rpm = (((int)buf[3] * 256) + buf[4]) / 4;
        return true;
      }
    }
  }
  return false;
}

// --- Function: Read DTC Codes via Mode 03 ---
// Sends a Mode 03 request and parses the response.
// It stores up to maxCodes DTC codes in dtcCodes and returns true if at least one is found.
bool getDTC(uint16_t *dtcCodes, byte maxCodes, byte &codeCount)
{
  byte data[8] = {0x02, 0x03, 0, 0, 0, 0, 0, 0};
  CAN.sendMsgBuf(0x7DF, 0, 8, data);
  unsigned long timeout = millis();
  codeCount = 0;
  while (millis() - timeout < 1000)
  {
    if (CAN_MSGAVAIL == CAN.checkReceive())
    {
      byte len = 0;
      byte buf[8];
      CAN.readMsgBuf(&len, buf);
      // Expect a positive response for Mode 03: first data byte should be 0x43.
      if (buf[1] == 0x43 || buf[1] == 0x5A)
      {

        // Starting at byte 2, every two bytes represent a DTC.
        for (int i = 2; i + 1 < len && codeCount < maxCodes; i += 2)
        {
          uint16_t dtc = (buf[i] << 8) | buf[i + 1];
          if (dtc == 0)
            continue;
          dtcCodes[codeCount++] = dtc;
        }
        break;
      }
    }
  }
  return (codeCount > 0);
}

// --- Get Warning Status and Build Sensor Groups ---
// Reads Mode 01 PID 0x01 to check MIL status and DTC count.
// If MIL is on, retrieves DTC codes via getDTC() and classifies them.
// For each active group, adds the full sensor list from our mapping.
void getWarningStatus(JSONVar &jsonObj)
{
  // Send a PID 0x01 request for MIL status and DTC count.
  byte data[8] = {0x02, 0x01, 0x01, 0, 0, 0, 0, 0};
  CAN.sendMsgBuf(0x7DF, 0, 8, data);
  unsigned long timeout = millis();
  int value = 0;
  bool gotPID01 = false;
  while (millis() - timeout < 1000)
  {
    if (CAN_MSGAVAIL == CAN.checkReceive())
    {
      byte len = 0;
      byte buf[8];
      CAN.readMsgBuf(&len, buf);
      if (buf[1] == 0x41 && buf[2] == 0x01)
      {
        value = buf[3]; // Assume first data byte carries MIL status & DTC count
        gotPID01 = true;
        break;
      }
    }
  }
  if (!gotPID01)
    return;

  bool milOn = (value & 0x80) != 0;
  jsonObj["check_engine_on"] = milOn;
  jsonObj["num_dtc_codes"] = value & 0x7F;

  if (milOn)
  {
    uint16_t codes[8];
    byte numCodes = 0;
    if (getDTC(codes, 8, numCodes))
    {
      char rawDTCs[128] = "";
      bool first = true;
      bool activeGroup[4] = {false, false, false, false}; // order: check_engine, transmission, abs, airbag

      for (byte i = 0; i < numCodes; i++)
      {
        char dtcStr[6];
        formatDTC(codes[i], dtcStr, sizeof(dtcStr));
        if (!first)
        {
          strncat(rawDTCs, ",", sizeof(rawDTCs) - strlen(rawDTCs) - 1);
        }
        strncat(rawDTCs, dtcStr, sizeof(rawDTCs) - strlen(rawDTCs) - 1);
        first = false;
        const char *system = classifyDTC(dtcStr);
        if (system)
        {
          for (int j = 0; j < numGroups; j++)
          {
            if (strcmp(groups[j].system, system) == 0)
            {
              activeGroup[j] = true;
            }
          }
        }
      }
      jsonObj["dtc_codes"] = String(rawDTCs);

      JSONVar warningLights = JSON.parse("{}");
      for (int j = 0; j < numGroups; j++)
      {
        if (activeGroup[j])
        {
          JSONVar groupObj = JSON.parse("{}");
          JSONVar sensors = JSON.parse("[]");
          for (byte k = 0; k < groups[j].sensorCount; k++)
          {
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

// --- Build and Print JSON Data ---
void sendJsonData()
{
  JSONVar jsonObj = JSON.parse("{}");
  jsonObj["timestamp"] = millis();

  int speed = 0, rpm = 0;
  if (getSpeed(&speed))
    jsonObj["speed"] = speed;
  else
    jsonObj["speed"] = -1;

  if (getRPM(&rpm))
    jsonObj["rpm"] = rpm;
  else
    jsonObj["rpm"] = -1;

  getWarningStatus(jsonObj);

  String jsonString = JSON.stringify(jsonObj);
  Serial.println(jsonString);
}

// --- Set CAN Masks and Filters ---
void set_mask_filt()
{
  CAN.init_Mask(0, 0, 0x7FC);
  CAN.init_Mask(1, 0, 0x7FC);
  CAN.init_Filt(0, 0, 0x7E8);
  CAN.init_Filt(1, 0, 0x7E8);
  CAN.init_Filt(2, 0, 0x7E8);
  CAN.init_Filt(3, 0, 0x7E8);
  CAN.init_Filt(4, 0, 0x7E8);
  CAN.init_Filt(5, 0, 0x7E8);
}

void setup()
{
  Serial.begin(115200);
  while (!Serial)
    ;

  // For OBD-II GPS Dev Kit RP2040 version
  pinMode(12, OUTPUT);
  digitalWrite(12, HIGH);
  // Initialize CAN bus at 500 kbps like your working code
  while (CAN_OK != CAN.begin(CAN_500KBPS))
  {
    Serial.println("CAN init fail, retry...");
    delay(100);
  }
  Serial.println("CAN init ok!");

  // Set mask and filters to receive OBD-II responses
  set_mask_filt();

  // Give some time for the bus to settle
  delay(1000);
}

void loop()
{
  sendJsonData();
  delay(1000);
}

// #include <SPI.h>
// #include <mcp_canbus.h>

// // --- CAN Bus Setup ---
// #define SPI_CS_PIN  9
// MCP_CAN CAN(SPI_CS_PIN);

// void set_mask_filt() {
//   CAN.init_Mask(0, 0, 0x7FC);
//   CAN.init_Mask(1, 0, 0x7FC);
//   CAN.init_Filt(0, 0, 0x7E8);
//   CAN.init_Filt(1, 0, 0x7E8);
//   CAN.init_Filt(2, 0, 0x7E8);
//   CAN.init_Filt(3, 0, 0x7E8);
//   CAN.init_Filt(4, 0, 0x7E8);
//   CAN.init_Filt(5, 0, 0x7E8);
// }

// void setup() {
//   Serial.begin(115200);
//   while (!Serial);

//   // For OBD-II GPS Dev Kit RP2040 version
//   pinMode(12, OUTPUT);
//   digitalWrite(12, HIGH);

//   // Initialize CAN bus at 500 kbps
//   while (CAN_OK != CAN.begin(CAN_500KBPS)) {
//     Serial.println("CAN init fail, retry...");
//     delay(100);
//   }
//   Serial.println("CAN init ok!");

//   // Set mask and filters to receive OBD-II responses
//   set_mask_filt();

//   // Give some time for the bus to settle
//   delay(1000);
// }

// void loop() {
//   // Send a Mode 03 request to get DTCs
//   byte request[8] = {0x02, 0x03, 0, 0, 0, 0, 0, 0};
//   CAN.sendMsgBuf(0x7DF, 0, 8, request);
//   Serial.println("Sent Mode 03 request");

//   unsigned long startTime = millis();
//   // Read responses for 1 second
//   while (millis() - startTime < 1000) {
//     if (CAN_MSGAVAIL == CAN.checkReceive()) {
//       byte len = 0;
//       byte buf[8];
//       CAN.readMsgBuf(&len, buf);
//       Serial.print("Received: ");
//       for (int i = 0; i < len; i++) {
//         // Print each byte in hexadecimal format
//         if (buf[i] < 0x10) Serial.print("0");
//         Serial.print(buf[i], HEX);
//         Serial.print(" ");
//       }
//       Serial.println();
//     }
//   }
//   Serial.println("---- End of Response ----");
//   delay(2000); // wait 2 seconds before next request
// }
