#include <WiFi.h>
#include <ArduinoJson.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>

// const char *ssid = "Aaron iPhone 15";
// const char *password = "12345678";
// const char* ssid = "Naumaan iPhone";
// const char* password = "joinup123";
const char *ssid = "Sheikh-Deco";
const char *password = "9058477273";
const int UART_BAUD = 9600;
const int RX2_PIN = 16;
const int TX2_PIN = 17;

WiFiClientSecure client;
HTTPClient http;

void sendJsonToServer(const String &jsonData)
{
  if (WiFi.status() == WL_CONNECTED)
  {
    static bool http_initialized = false;

    if (!http_initialized)
    {
      client.setInsecure();
      http.begin(client, "https://ghastly-singular-snake.ngrok.app/receive_obd_data");
      http.addHeader("Content-Type", "application/json");
      http_initialized = true;
    }

    // Send asynchronously - don't wait for response
    http.sendRequest("POST", jsonData.c_str());
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
}