export interface VehicleDisplayError {
    description: string
    OBDCode: string
    vehicleID: string
    timestamp: string
}

export interface VehicleHealthListProps {
  vehicleErrors: VehicleDisplayError[]
  vehicleSelected?: boolean
}

export const vehicleErrorData: VehicleDisplayError[] = [
  { "description": "Coolant Thermostat Circuit High", "OBDCode": "P0128", "vehicleID": "123", "timestamp": "2024-07-10T10:00:00Z" },
  { "description": "O2 Sensor Circuit Slow Response", "OBDCode": "P0133", "vehicleID": "124", "timestamp": "2024-07-10T10:05:00Z" },
  { "description": "Cylinder 1 Misfire Detected", "OBDCode": "P0301", "vehicleID": "125", "timestamp": "2024-07-10T10:10:00Z" },
  { "description": "Catalyst System Efficiency Below Threshold", "OBDCode": "P0420", "vehicleID": "126", "timestamp": "2024-07-10T10:15:00Z" },
  { "description": "Evaporative Emission Control System Leak Detected", "OBDCode": "P0455", "vehicleID": "127", "timestamp": "2024-07-10T10:20:00Z" },
  { "description": "Mass or Volume Air Flow Circuit Low Input", "OBDCode": "P0102", "vehicleID": "128", "timestamp": "2024-07-10T10:25:00Z" },
  { "description": "Throttle/Pedal Position Sensor/Switch A Circuit Range/Performance Problem", "OBDCode": "P0121", "vehicleID": "129", "timestamp": "2024-07-10T10:30:00Z" },
  { "description": "Intake Air Temperature Circuit High Input", "OBDCode": "P0113", "vehicleID": "130", "timestamp": "2024-07-10T10:35:00Z" },
  { "description": "Knock Sensor 1 Circuit Malfunction", "OBDCode": "P0325", "vehicleID": "131", "timestamp": "2024-07-10T10:40:00Z" },
  { "description": "Fuel Temperature Sensor A Circuit High Input", "OBDCode": "P0183", "vehicleID": "132", "timestamp": "2024-07-10T10:45:00Z" }
]
