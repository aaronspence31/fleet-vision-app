#include <WiFi.h>
#include <ArduinoJson.h>
#include <HTTPClient.h>

// const char *ssid = "Aaron iPhone 15";
// const char *password = "12345678";
const char *ssid = "SPIDER-LAN";
const char *password = "GreatPowerGreatResponsibility123!";
const int UART_BAUD = 9600;
const int RX2_PIN = 16;
const int TX2_PIN = 17;

void sendJsonToServer(const String &jsonData)
{
  if (WiFi.status() == WL_CONNECTED)
  {
    HTTPClient http;
    // http.begin("http://172.20.10.14:5000/obd_data");
    // http.begin("http://192.168.68.58:5000/obd_data");
    http.begin("http://192.168.68.56:5000/obd_data");

    http.addHeader("Content-Type", "application/json");

    int httpResponseCode = http.POST(jsonData);

    if (httpResponseCode > 0)
    {
      Serial.printf("HTTP Response code: %d\n", httpResponseCode);
    }
    else
    {
      Serial.printf("Error code: %d\n", httpResponseCode);
    }

    http.end();
  }
}

void setup()
{
  Serial.begin(115200);
  Serial2.begin(UART_BAUD, SERIAL_8N1, RX2_PIN, TX2_PIN);

  Serial.println("connecting to wifi...");

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");
  Serial.printf("IP Address: %s\n", WiFi.localIP().toString().c_str());
  Serial.printf("Gateway IP: %s\n", WiFi.gatewayIP().toString().c_str());
}

void loop()
{
  static String jsonBuffer;

  while (Serial2.available())
  {
    char c = Serial2.read();

    if (c == '\n')
    {
      Serial.println(jsonBuffer);
      StaticJsonDocument<200> doc;
      DeserializationError error = deserializeJson(doc, jsonBuffer);

      if (!error)
      {
        sendJsonToServer(jsonBuffer);
      }

      jsonBuffer = "";
    }
    else
    {
      jsonBuffer += c;
    }
  }
  delay(10);
}
