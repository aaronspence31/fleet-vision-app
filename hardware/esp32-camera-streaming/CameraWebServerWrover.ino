#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

#define CAMERA_MODEL_WROVER_KIT

#include "camera_pins.h"

// const char *ssid = "Aaron iPhone 15";
// const char *password = "12345678";
const char *ssid = "SPIDER-LAN";
const char *password = "GreatPowerGreatResponsibility123!";

WebServer server(80);

// Set the initial resolution
framesize_t currentResolution = FRAMESIZE_HD;

void setResolution(framesize_t resolution)
{
  sensor_t *s = esp_camera_sensor_get();
  s->set_framesize(s, resolution);
  currentResolution = resolution;
}

void handleRoot()
{
  String html = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ESP32-CAM Video Stream</title>
<script>
function changeResolution(resolution) {
  fetch('/resolution?set=' + resolution)
    .then(response => response.text())
    .then(data => {
      console.log(data);
      location.reload();
    });
}
</script>
</head>
<body>
<h1>ESP32-CAM Video Stream</h1>
<img src="/stream" style="width: 100%;">
<div>
<button onclick="changeResolution('QVGA')">320x240</button>
<button onclick="changeResolution('VGA')">640x480</button>
<button onclick="changeResolution('SVGA')">800x600</button>
<button onclick="changeResolution('XGA')">1024x768</button>
<button onclick="changeResolution('SXGA')">1280x1024</button>
</div>
</body>
</html>
)rawliteral";
  server.send(200, "text/html", html);
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

    if (!client.connected())
    {
      send = false;
    }
  }
}

void handleResolution()
{
  String resolution = server.arg("set");

  if (resolution == "QVGA")
    setResolution(FRAMESIZE_QVGA);
  else if (resolution == "VGA")
    setResolution(FRAMESIZE_VGA);
  else if (resolution == "SVGA")
    setResolution(FRAMESIZE_SVGA);
  else if (resolution == "XGA")
    setResolution(FRAMESIZE_XGA);
  else if (resolution == "SXGA")
    setResolution(FRAMESIZE_SXGA);

  server.send(200, "text/plain", "Resolution set to " + resolution);
}

void setup()
{
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

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
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK)
  {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");

  server.on("/", handleRoot);
  server.on("/stream", HTTP_GET, handleStream);
  server.on("/resolution", HTTP_GET, handleResolution);

  server.begin();
  Serial.println("HTTP server started");

  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");
}

void loop()
{
  server.handleClient();
}