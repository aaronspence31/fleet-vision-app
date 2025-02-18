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
  while (millis() - timeout < 400)
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
  while (millis() - timeout < 400)
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

// --- Get Warning Status ---
// Sends a Mode 01 request to get the MIL status and number of DTCs.
// If no valid response is received, sets check_engine_on to null and num_dtc_codes to -1.
void getWarningStatus(JSONVar &jsonObj)
{
  // Send a PID 0x01 request for MIL status & DTC count.
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
  {
    jsonObj["check_engine_on"] = nullptr; // using null to indicate unavailability
    jsonObj["num_dtc_codes"] = -1;
    return;
  }

  bool milOn = (value & 0x80) != 0;
  jsonObj["check_engine_on"] = milOn;
  jsonObj["num_dtc_codes"] = value & 0x7F;
}

// --- Build and Print JSON Data ---
void sendJsonData()
{
  JSONVar jsonObj = JSON.parse("{}");
  // jsonObj["timestamp"] = millis();

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
  delay(400);
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
//   while (millis() - startTime < 7000) {
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
