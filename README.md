# Fleet Vision

![Fleet Vision Logo](web-app/public/images/fleet-vision-logo.png)

## Overview

Fleet Vision is a comprehensive vehicle monitoring and driver safety system designed to enhance fleet management and driver safety through real-time monitoring and analytics. The system combines computer vision, OBD-II vehicle data, and cloud-based analytics to provide a complete picture of driver behavior and vehicle performance.

## Demo Highlights

### Video Demonstration

[![Fleet Vision Demo](thumbnail.png)](https://drive.google.com/file/d/1jhZySFJ_BmHR1zjNEgbxbeXLgKqwypyG/view?usp=sharing)

_Click the image above to watch the full demo video_

### System in Action

![Dashboard Overview]

_[TODO: Add an impressive screenshot of the main dashboard in action]_

![Real-time Monitoring]

_[TODO: Add a screenshot showing real-time driver monitoring with analytics]_

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Hardware Components](#hardware-components)
- [Software Components](#software-components)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Detailed Demos](#detailed-demos)

## Features

### Real-time Driver Monitoring

- **Facial Analysis**: Detects driver drowsiness by monitoring eye closure and yawning
- **Body Posture Analysis**: Monitors driver posture and behavior for distracted driving
- **OBD-II Data Collection**: Captures real-time vehicle data including speed, RPM, and more

### Analytics Dashboard

- **Safety Score**: Comprehensive driver safety scoring based on multiple factors
- **Trip History**: Detailed logs of all driving sessions with timestamps
- **Performance Metrics**: Vehicle performance data visualization
- **Incident Detection**: Automatic detection and logging of safety incidents

### System Integration

- **Cloud Storage**: All data securely stored in Firebase
- **Real-time Processing**: Edge computing on ESP32 devices with server-side processing
- **Web Interface**: Modern, responsive dashboard for fleet managers

## System Architecture

![System Architecture Diagram]

_[TODO: Add system architecture diagram showing the flow of data from hardware devices through the server to the web application]_

The Fleet Vision system consists of three main components:

1. **Hardware Devices**: ESP32 cameras and OBD-II interface connected to the vehicle
2. **Real-time Server**: Python Flask server that processes data streams from the hardware
3. **Web Application**: Next.js dashboard for visualizing and analyzing the data

Data flows from the vehicle through the hardware devices to the real-time server, which processes the data and stores it in Firebase. The web application retrieves the data from Firebase and presents it to the user. The web application also recieves the classified data from the real-time server via Server Sent Events and displays it on its live feed pages.

## Hardware Components

### ESP32 Camera Modules

- **Face Camera**: ESP32-CAM AI-Thinker module with OV2640 camera for monitoring the driver's face
- **Body Camera**: ESP32-CAM AI-Thinker module with OV2640 camera for monitoring the driver's body posture

### OBD-II Interface

- **ESP32 Microcontroller**: Connected to the CAN bus via MCP2515 CAN controller
- **Longan Labs OBD-2 Dev Kit**: Used as the OBD-II interface for the vehicle

![Hardware Setup in Vehicle](hardware_setup_in_vehicle.png)

_The image above shows the ESP32 cameras and OBD-II interface installed in a vehicle_

## Software Components

### Real-time Server

- **Flask Backend**: Handles data streams from the hardware devices
- **Computer Vision Models**: Processes camera feeds for drowsiness and distraction detection
- **Data Processing**: Analyzes OBD-II data for vehicle performance metrics
- **Firebase Integration**: Stores processed data in Firebase

### Web Application

- **Next.js Frontend**: Modern, responsive dashboard
- **Material UI**: Clean, intuitive user interface
- **Real-time Updates**: Live data visualization
- **Historical Analysis**: Trip history and performance trends

## Installation and Setup

### Prerequisites

- Node.js (v14 or higher)
- Python 3.8 or higher
- Firebase account
- ngrok account (for remote access to the real-time server)
- Arduino IDE with ESP32 board support
- ESP32-CAM AI-Thinker modules (2x)
- Longan Labs OBD-2 Dev Kit
- Jumper wires and breadboard

### Web Application Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/fleet-vision-app.git
   cd fleet-vision-app
   ```

2. Install web application dependencies:

   ```bash
   cd web-app
   npm install
   ```

3. Create a `.env` file in the `web-app` directory with your Firebase configuration:

   ```
   NEXT_PUBLIC_FIREBASE_API_KEY=your_api_key
   NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your_auth_domain
   NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_project_id
   NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your_storage_bucket
   NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your_messaging_sender_id
   NEXT_PUBLIC_FIREBASE_APP_ID=your_app_id
   ```

4. Start the web application:
   ```bash
   npm run dev
   ```

### Real-time Server Setup

1. Install server dependencies:

   ```bash
   cd realtime-server
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the `realtime-server` directory with your ngrok configuration:

   ```
   NGROK_AUTH_TOKEN=your_ngrok_auth_token
   ```

3. Add your Firebase service account key file as `key.json` in the `realtime-server` directory.

4. Start the real-time server:
   ```bash
   python server.py
   ```

### Hardware Setup

#### ESP32 Camera Setup

1. **Hardware Requirements**:

   - 2x ESP32-CAM AI-Thinker modules with OV2640 camera
   - FTDI programmer or CP2102 USB-to-TTL converter for uploading code
   - Jumper wires
   - 5V power supply

2. **Wiring the ESP32-CAM for Programming**:

   - Connect the ESP32-CAM to the FTDI programmer:
     - ESP32-CAM GND → FTDI GND
     - ESP32-CAM 5V/VCC → FTDI VCC (5V)
     - ESP32-CAM U0R (GPIO3) → FTDI TX
     - ESP32-CAM U0T (GPIO1) → FTDI RX
     - Connect GPIO0 to GND (only during programming)
   - After uploading, disconnect GPIO0 from GND for normal operation

3. **Installing the Code**:

   - Open the Arduino IDE with ESP32 board support installed
   - Select "AI Thinker ESP32-CAM" from the boards menu
   - For the face monitoring camera:
     Open `hardware/esp32-camera-streaming/CameraWebServerFace.ino` in the Arduino IDE
   - For the body posture monitoring camera:
     Open `hardware/esp32-camera-streaming/CameraWebServerBody.ino` in the Arduino IDE

4. **Configure WiFi Settings**:

   - In each `.ino` file, update the WiFi credentials:
     ```cpp
     const char *ssid = "Your_WiFi_SSID";
     const char *password = "Your_WiFi_Password";
     ```

5. **Upload the Code**:

   - Connect GPIO0 to GND
   - Press the reset button on the ESP32-CAM
   - Upload the code
   - Disconnect GPIO0 from GND
   - Press reset again to start normal operation

6. **Mounting the Cameras**:
   - Face Camera: Mount facing the driver's face, ideally on the dashboard or A-pillar
   - Body Camera: Mount with a wider view of the driver's upper body, ideally on the dashboard or center console

#### OBD-II Interface Setup

1. **Hardware Requirements**:

   - ESP32 development board
   - MCP2515 CAN Bus module
   - OBD-II to DB9 cable
   - Jumper wires
   - Breadboard

2. **Wiring the CAN Bus Module**:

   - Connect the MCP2515 to the ESP32:
     - MCP2515 CS → ESP32 GPIO9 (as defined in the code: `#define SPI_CS_PIN 9`)
     - MCP2515 SO (MISO) → ESP32 MISO
     - MCP2515 SI (MOSI) → ESP32 MOSI
     - MCP2515 SCK → ESP32 SCK
     - MCP2515 VCC → ESP32 3.3V
     - MCP2515 GND → ESP32 GND

3. **Installing the Code**:

   - Open the Arduino IDE
   - Select your ESP32 board from the boards menu
   - Open `hardware/obd-streaming/FinalOBD.ino` in the Arduino IDE

4. **Upload the Code**:

   - Connect the ESP32 to your computer
   - Upload the code

5. **Connecting to the Vehicle**:

   - Connect the OBD-II to DB9 cable to your vehicle's OBD-II port
   - Connect the DB9 end to the CAN Bus module
   - Power on the ESP32 (can be powered via USB or an external 5V supply)

6. **Testing the Connection**:
   - Start your vehicle
   - The ESP32 should connect to your WiFi network
   - The real-time server should receive OBD-II data

## Usage

### Starting a Monitoring Session

1. Start the real-time server.
2. Power on the ESP32 cameras and OBD-II interface.
3. Open the web application.
4. Navigate to the dashboard to view real-time data.

### Viewing Historical Data

1. Open the web application.
2. Navigate to the "Sessions" section.
3. Select a session to view detailed data.

### Analyzing Driver Performance

1. Open the web application.
2. Navigate to the "Analytics" section.
3. View safety scores, performance metrics, and incident reports.

## Detailed Demos

### Main Dashboard

![Dashboard]

_[TODO: Add screenshot of the main dashboard]_

The main dashboard provides an overview of all vehicle and driver metrics, including:

- Current safety score
- Recent driving sessions
- Alert history
- Performance trends

### Vehicle Information Interface

![Vehicle Info]

_[TODO: Add screenshot of the vehicle info page]_

The Vehicle Information interface displays:

- Real-time speed and RPM gauges
- Engine performance metrics
- Trip distance and duration
- Fuel efficiency data

### Face Monitoring Demo

![Face Monitoring]

_[TODO: Add screenshot of the face monitoring interface]_

The Face Monitoring system:

- Detects driver drowsiness through eye closure detection
- Monitors yawning frequency
- Provides real-time alerts for drowsy driving
- Records incidents for later review

**Video Demo of Face Monitoring:**

[![Face Monitoring Demo](https://img.youtube.com/vi/1jhZySFJ_BmH/0.jpg)](https://drive.google.com/file/d/1jhZySFJ_BmHR1zjNEgbxbeXLgKqwypyG/view?usp=sharing)

_Click the image above to watch the face monitoring demo_

### Body Posture Monitoring Demo

![Body Posture Monitoring Interface](https://i.imgur.com/Ij9Yvqm.jpg)

The Body Posture Monitoring system:

- Detects distracted driving behaviors
- Monitors driver position and movement
- Identifies unsafe postures
- Alerts when the driver is not focused on the road

**Video Demo of Body Posture Monitoring:**

[![Body Posture Monitoring Demo](https://i.imgur.com/Ij9Yvqm.jpg)](https://drive.google.com/file/d/1jhZySFJ_BmHR1zjNEgbxbeXLgKqwypyG/view?usp=sharing)

_Click the image above to watch the body posture monitoring demo_

### OBD-II Data Monitoring Demo

![OBD-II Monitoring]

_[TODO: Add screenshot of the OBD-II monitoring interface]_

The OBD-II Monitoring system:

- Captures real-time vehicle telemetry
- Monitors engine performance parameters
- Detects potential mechanical issues
- Tracks fuel consumption and efficiency

**Video Demo of OBD-II Monitoring:**

[![OBD-II Monitoring Demo](https://img.youtube.com/vi/1jhZySFJ_BmH/0.jpg)](https://drive.google.com/file/d/1jhZySFJ_BmHR1zjNEgbxbeXLgKqwypyG/view?usp=sharing)

_Click the image above to watch the OBD-II monitoring demo_

---

## Note to Project Owner

To complete this README, please:

1. Add the system architecture diagram showing data flow between components
2. Add screenshots of all interfaces (dashboard, vehicle info, face monitoring, body monitoring, OBD-II monitoring)
3. Record and add video demonstrations for each monitoring system
4. Update any missing configuration details or instructions

These visual elements will greatly enhance the documentation and make it easier for users to understand the system.
