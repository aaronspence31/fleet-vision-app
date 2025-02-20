#include <WiFi.h>
#include <ArduinoJson.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>

const char *ssid = "Sheikh-Deco";
const char *password = "9058477273";
// const char* ssid = "Naumaan iPhone";
// const char* password = "joinup123";
const int UART_BAUD = 9600;
// const int RX2_PIN = 16;
// const int TX2_PIN = 17;
const int RX_PIN = 3; // For antenna ESP Wroom 32U
const int TX_PIN = 1; // For antenna ESP Wroom 32U

// for speed test
const int SPEED_TEST_ITERATIONS = 5;                                                      // Number of test packets to send
const int TEST_PACKET_SIZE = 1024;                                                        // Size of each test packet in bytes
const char *SPEED_TEST_URL = "https://ghastly-singular-snake.ngrok.app/receive_obd_data"; // Use your existing endpoint

struct NetworkMetrics
{
    float upload_speed_kbps;
    int32_t rssi;
    String wifi_strength;
    bool is_network_suitable;
};

// Performance monitoring
const int MAX_SAMPLES = 100;
struct Timing
{
    uint32_t receive_micros; // When UART data was received
    uint32_t parse_micros;   // When JSON parsing completed
    uint32_t send_micros;    // When HTTP request sent
    size_t data_size;        // Size of JSON data
};

Timing metrics[MAX_SAMPLES];
int sample_count = 0;
bool test_running = false;

WiFiClientSecure client;
HTTPClient http;

void printMetrics()
{
    unsigned long total_parse_time = 0;
    unsigned long total_send_time = 0;
    unsigned long total_process_time = 0;
    size_t total_size = 0;

    // Calculate intervals between frames
    unsigned long total_frame_interval = 0;
    int interval_count = 0;

    for (int i = 0; i < sample_count; i++)
    {
        unsigned long parse_time = (metrics[i].parse_micros - metrics[i].receive_micros) / 1000;
        unsigned long send_time = (metrics[i].send_micros - metrics[i].parse_micros) / 1000;
        unsigned long process_time = (metrics[i].send_micros - metrics[i].receive_micros) / 1000;

        total_parse_time += parse_time;
        total_send_time += send_time;
        total_process_time += process_time;
        total_size += metrics[i].data_size;

        if (i > 0)
        {
            unsigned long frame_interval = (metrics[i].receive_micros - metrics[i - 1].receive_micros) / 1000;
            total_frame_interval += frame_interval;
            interval_count++;
        }
    }

    float avg_parse_time = total_parse_time / (float)sample_count;
    float avg_send_time = total_send_time / (float)sample_count;
    float avg_process_time = total_process_time / (float)sample_count;
    float avg_size = total_size / (float)sample_count;
    float avg_frame_interval = interval_count > 0 ? total_frame_interval / (float)interval_count : 0;

    float actual_fps = avg_frame_interval > 0 ? 1000.0 / avg_frame_interval : 0;
    float theoretical_fps = avg_process_time > 0 ? 1000.0 / avg_process_time : 0;

    Serial.println("\n=== Performance Test Results ===");
    Serial.printf("Samples collected: %d\n", sample_count);
    Serial.printf("Average parse time: %.2f ms\n", avg_parse_time);
    Serial.printf("Average send prep time: %.2f ms\n", avg_send_time);
    Serial.printf("Average total processing time: %.2f ms\n", avg_process_time);
    Serial.printf("Average frame interval: %.2f ms\n", avg_frame_interval);
    Serial.printf("Average frame size: %.1f bytes\n", avg_size);
    Serial.printf("Actual frame rate: %.2f fps\n", actual_fps);
    Serial.printf("Theoretical max frame rate: %.2f fps\n", theoretical_fps);
    Serial.printf("Data throughput: %.2f bytes/sec\n", avg_size * actual_fps);
    Serial.println("===============================\n");
}

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

        Serial.println(jsonData);
    }
}

/*
 *               Speed test functions
 */

// Function to generate test data of specified size
String generateTestData(int size)
{
    StaticJsonDocument<1500> doc; // Adjust size if needed

    // Create a JSON object with random data to simulate OBD data
    doc["test_type"] = "speed_test";
    doc["packet_size"] = size;

    // Fill with random data to reach desired size
    String random_data;
    random_data.reserve(size);
    const char charset[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    while (random_data.length() < size)
    {
        random_data += charset[random(0, sizeof(charset) - 1)];
    }

    doc["data"] = random_data;

    String jsonString;
    serializeJson(doc, jsonString);
    return jsonString;
}

// Function to get WiFi signal strength description
String getWiFiStrength(int32_t rssi)
{
    if (rssi >= -50)
        return "Excellent";
    else if (rssi >= -60)
        return "Good";
    else if (rssi >= -70)
        return "Fair";
    else if (rssi >= -80)
        return "Weak";
    else
        return "Very Weak";
}

NetworkMetrics testNetworkSpeed()
{
    NetworkMetrics metrics;
    WiFiClientSecure client;
    client.setInsecure();
    HTTPClient http;

    // Get RSSI (signal strength)
    metrics.rssi = WiFi.RSSI();
    metrics.wifi_strength = getWiFiStrength(metrics.rssi);

    Serial.println("\n=== Network Speed Test Starting ===");
    Serial.printf("WiFi Signal Strength: %d dBm (%s)\n", metrics.rssi, metrics.wifi_strength.c_str());

    // Generate test data
    String testData = generateTestData(TEST_PACKET_SIZE);
    unsigned long totalTime = 0;
    int successfulTests = 0;

    http.begin(client, SPEED_TEST_URL);
    http.addHeader("Content-Type", "application/json");

    for (int i = 0; i < SPEED_TEST_ITERATIONS; i++)
    {
        Serial.printf("Speed test iteration %d/%d...\n", i + 1, SPEED_TEST_ITERATIONS);

        unsigned long startTime = millis();
        int httpResponseCode = http.POST(testData);
        unsigned long endTime = millis();

        if (httpResponseCode == 200)
        {
            unsigned long duration = endTime - startTime;
            totalTime += duration;
            successfulTests++;

            // Calculate speed for this iteration
            float speed_kbps = (testData.length() * 8.0) / (duration); // Kilobits per second
            Serial.printf("Iteration %d: %.2f kbps\n", i + 1, speed_kbps);
        }
        else
        {
            Serial.printf("Iteration %d failed with code: %d\n", i + 1, httpResponseCode);
        }

        delay(100); // Short delay between tests
    }

    http.end();

    // Calculate average upload speed
    if (successfulTests > 0)
    {
        float avgTime = totalTime / (float)successfulTests;
        metrics.upload_speed_kbps = (TEST_PACKET_SIZE * 8.0) / avgTime; // Kilobits per second
    }
    else
    {
        metrics.upload_speed_kbps = 0;
    }

    // Determine if network is suitable for your application
    // Assuming minimum required speed is 5fps * 64 bytes * 8 bits = 2.56 kbps
    const float MIN_REQUIRED_SPEED = 2.56; // kbps
    metrics.is_network_suitable = metrics.upload_speed_kbps >= MIN_REQUIRED_SPEED;

    Serial.println("\n=== Network Test Results ===");
    Serial.printf("Average Upload Speed: %.2f kbps\n", metrics.upload_speed_kbps);
    Serial.printf("WiFi Signal Strength: %d dBm (%s)\n", metrics.rssi, metrics.wifi_strength.c_str());
    Serial.printf("Network Suitable for Application: %s\n", metrics.is_network_suitable ? "Yes" : "No");
    Serial.printf("Successful Tests: %d/%d\n", successfulTests, SPEED_TEST_ITERATIONS);
    Serial.println("=============================\n");

    return metrics;
}

// Function to add to your setup() before starting main performance test
void performNetworkCheck()
{
    // Wait for WiFi connection
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi Connected!");

    // Test network speed
    NetworkMetrics networkMetrics = testNetworkSpeed();

    // Warning if network might be too slow
    if (!networkMetrics.is_network_suitable)
    {
        Serial.println("WARNING: Network speed might be insufficient for desired frame rate!");
        Serial.println("Consider:");
        Serial.println("1. Moving closer to WiFi access point");
        Serial.println("2. Reducing data packet size");
        Serial.println("3. Lowering target frame rate");
    }

    // Short delay before starting main test
    delay(1000);
}

/*
 *           End of speed test functions
 */

void setup()
{
    Serial.begin(115200);
    // Serial2.begin(UART_BAUD, SERIAL_8N1, RX2_PIN, TX2_PIN);
    Serial1.begin(UART_BAUD, SERIAL_8N1, RX_PIN, TX_PIN); // For ESP Wroom 32U (antenna)

    Serial.println("Connecting to WiFi...");
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi Connected!");

    // Perform network speed test
    performNetworkCheck();

    // Start performance test
    test_running = true;
    Serial.println("Starting performance test...");
}

void loop()
{
    static String jsonBuffer;

    // while (Serial2.available()) {
    //     char c = Serial2.read();
    while (Serial1.available())
    {
        char c = Serial1.read();

        if (c == '\n')
        {
            if (test_running && sample_count < MAX_SAMPLES)
            {
                metrics[sample_count].receive_micros = micros();
                metrics[sample_count].data_size = jsonBuffer.length();
            }

            StaticJsonDocument<200> doc;
            DeserializationError error = deserializeJson(doc, jsonBuffer);

            if (test_running && sample_count < MAX_SAMPLES)
            {
                metrics[sample_count].parse_micros = micros();
            }

            if (!error)
            {
                if (test_running && sample_count < MAX_SAMPLES)
                {
                    sendJsonToServer(jsonBuffer);
                    metrics[sample_count].send_micros = micros();
                    sample_count++;

                    if (sample_count >= MAX_SAMPLES)
                    {
                        printMetrics();
                        test_running = false;
                    }
                }
                else
                {
                    sendJsonToServer(jsonBuffer);
                }
            }

            jsonBuffer = "";
        }
        else
        {
            jsonBuffer += c;
        }
    }
}