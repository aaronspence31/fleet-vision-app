#include "esp_camera.h"
#include <WiFi.h>
#include <ESPmDNS.h>
#include <WiFiClient.h>
#include <WebServer.h>

// Select camera model
#define CAMERA_MODEL_AI_THINKER // Has PSRAM

#include "camera_pins.h"

// const char *ssid = "Aaron iPhone 15";
// const char *password = "12345678";
const char *ssid = "SPIDER-LAN";
const char *password = "GreatPowerGreatResponsibility123!";

WebServer server(80);

// Set the initial resolution manually here
framesize_t currentResolution = FRAMESIZE_HD;

// Function to print FPS to the Serial Monitor
void printFPS()
{
    static int frameCount = 0;
    static unsigned long lastTime = 0;
    frameCount++;
    if (millis() - lastTime >= 1000)
    {
        Serial.printf("FPS: %d\n", frameCount);
        frameCount = 0;
        lastTime = millis();
    }
}

void handleRoot()
{
    server.send(200, "text/html", R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ESP32-CAM Video Stream</title>
</head>
<body>
<h1>ESP32-CAM Video Stream</h1>
<img src="/stream" style="width: 50%;">
</body>
</html>
)rawliteral");
}

void handleNotFound()
{
    server.send(404, "text/plain", "Not found");
}

void handleStream()
{
    WiFiClient client = server.client();
    camera_fb_t *fb = NULL;
    bool send = true;
    String response = "HTTP/1.1 200 OK\r\n" +
                      String("Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n");
    server.sendContent(response);

    while (send)
    {
        fb = esp_camera_fb_get();
        if (!fb)
        {
            Serial.println("Camera capture failed");
            server.send(500, "text/plain", "Camera capture failed");
            return;
        }

        response = "--frame\r\n" +
                   String("Content-Type: image/jpeg\r\n\r\n");
        server.sendContent(response);
        client.write(fb->buf, fb->len);
        server.sendContent("\r\n");
        esp_camera_fb_return(fb);

        printFPS();

        if (!client.connected())
        {
            send = false;
        }
    }
}

void startCamera()
{
    // Ensure the camera is deinitialized before reinitializing
    esp_camera_deinit();

    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.frame_size = currentResolution;
    config.pixel_format = PIXFORMAT_JPEG; // for streaming
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.jpeg_quality = 12;
    config.fb_count = 1;

    if (config.pixel_format == PIXFORMAT_JPEG)
    {
        if (psramFound())
        {
            config.jpeg_quality = 10;
            config.fb_count = 2;
            config.grab_mode = CAMERA_GRAB_LATEST;
        }
        else
        {
            config.frame_size = FRAMESIZE_SVGA;
            config.fb_location = CAMERA_FB_IN_DRAM;
        }
    }
    else
    {
        config.frame_size = FRAMESIZE_240X240;
    }

    // Camera init
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK)
    {
        Serial.printf("Camera init failed with error 0x%x", err);
        return;
    }

    sensor_t *s = esp_camera_sensor_get();
    if (s->id.PID == OV3660_PID)
    {
        s->set_vflip(s, 1);
        s->set_brightness(s, 1);
        s->set_saturation(s, -2);
    }

    s->set_framesize(s, currentResolution);

#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
    s->set_vflip(s, 1);
    s->set_hmirror(s, 1);
#endif

#if defined(CAMERA_MODEL_ESP32S3_EYE)
    s->set_vflip(s, 1);
#endif
}

void setup()
{
    Serial.begin(115200);
    Serial.setDebugOutput(true);
    Serial.println();

    WiFi.begin(ssid, password);
    WiFi.setSleep(false);

    Serial.println("Connecting to WiFi...");
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(1000);
        Serial.print(".");
    }

    Serial.println("");
    Serial.println("WiFi connected");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());

    if (MDNS.begin("esp32body"))
    {
        Serial.println("MDNS responder started");
    }

    server.on("/", handleRoot);
    server.on("/stream", HTTP_GET, handleStream);
    server.onNotFound(handleNotFound);

    server.begin();
    Serial.println("HTTP server started");

    Serial.print("Camera Ready! Use 'http://");
    Serial.print(WiFi.localIP());
    Serial.println("' to connect");

    startCamera(); // Start camera initially
}

void loop()
{
    server.handleClient();
}