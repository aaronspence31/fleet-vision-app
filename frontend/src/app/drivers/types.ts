export interface Driver {
    name: string
    id: string
    safetyScore: number
    currentVehicle: string
    vehicleId: string
    status: 'Online' | 'Offline'
}

interface DriverColumns {
    field: string
    headerName: string
    flex: number
    headerClassName: string
}

export const driverColumns: DriverColumns[] = [
    {
      field: "name",
      headerName: "Name",
      flex: 1,
      headerClassName: 'header'
    },
    {
      field: "safetyScore",
      headerName: "Safety Score",
      flex: 1,
      headerClassName: 'header'
    },
    {
      field: "currentVehicle",
      headerName: "Vehicle",
      flex: 1,
      headerClassName: 'header'
    },
    {
      field: "status",
      headerName: "Status",
      flex: 1,
      headerClassName: 'header'
    },
  ]

export const driverData: Driver[] = [{
    "name": "John Jordan",
    "id": "8f3e4b2e-9a2f-47d1-8d2a-5e7a1c3b9f4d",
    "safetyScore": 0,
    "currentVehicle": "Toyota Camry",
    "vehicleId": "7a06c4b6-6e1f-4b48-bd22-68b66c1a5df3",
    "status": "Online"
  },
  {
    "name": "Casey Alex",
    "id": "7b9d2a3c-8e4f-47a2-9b5d-3e6d1a7c8f2b",
    "safetyScore": 0,
    "currentVehicle": "Ford F-150",
    "vehicleId": "2c5d0d9c-75b8-4cbf-bf39-5d456b6f3f9e",
    "status": "Online"
  },
  {
    "name": "Taylor Jordan",
    "id": "6a8e4b7f-2c5d-4a9b-8f3a-4d7e2c1b5a6d",
    "safetyScore": 0,
    "currentVehicle": "Honda Accord",
    "vehicleId": "865d4cd7-4904-4a1a-8d72-f2c4b2b75db4",
    "status": "Offline"
  },
  {
    "name": "Morgan Pat",
    "id": "5e4a9c3b-7d2f-4a1f-9b7d-6e2c4b1a5f3d",
    "safetyScore": 0,
    "currentVehicle": "Chevrolet Silverado",
    "vehicleId": "7a7e618e-204a-4038-bc45-6f65a3d9d517",
    "status": "Online"
  },
  {
    "name": "Chris Emily",
    "id": "4c7e2d5a-9b4f-47a1-8d7f-2e6a3b4c1f9d",
    "safetyScore": 0,
    "currentVehicle": "Nissan Altima",
    "vehicleId": "8f5a9c7f-224b-4969-bc43-4d7e48b85d68",
    "status": "Offline"
  },
  {
    "name": "Emily Pat",
    "id": "3b2d9c7e-4a6f-47d1-9a5b-7e2c4f1b6a3d",
    "safetyScore": 0,
    "currentVehicle": "Ford F-150",
    "vehicleId": "b2d9f6ae-4a32-4a21-a3fb-cbdd6f25f9c3",
    "status": "Online"
  },
  {
    "name": "Casey Taylor",
    "id": "2c7a4e9d-5b3f-48d1-9c6a-3e5a2b7d4f1a",
    "safetyScore": 0,
    "currentVehicle": "Toyota Camry",
    "vehicleId": "3b2e0b9c-85a4-4b1f-847f-77d0f6c4b1f4",
    "status": "Offline"
  },
  {
    "name": "Taylor John",
    "id": "1d8e5c3a-7f4b-48d1-8b7a-6e2f4c1b9a5d",
    "safetyScore": 0,
    "currentVehicle": "Chevrolet Silverado",
    "vehicleId": "b0c6d6e7-49db-45e7-b38f-946d9b5e2c58",
    "status": "Online"
  },
  {
    "name": "Morgan Jane",
    "id": "0a7c5e9b-2d4f-49a1-8b6f-7e3d2c4a5b9d",
    "safetyScore": 0,
    "currentVehicle": "Honda Accord",
    "vehicleId": "a3f8b6e5-29e8-40f6-98bb-5e4c6e5b1d48",
    "status": "Online"
  },
  {
    "name": "Pat Casey",
    "id": "9b8e3c7a-5d4f-49d1-9b7a-4e6f2a1b8c5d",
    "safetyScore": 0,
    "currentVehicle": "Ford F-150",
    "vehicleId": "0e8d0b6a-785a-40d4-9f1a-f0d6e4b2c9b6",
    "status": "Offline"
  },
  {
    "name": "Jordan Morgan",
    "id": "8a6e4d9c-2b5f-49a1-9d7a-5e2f3c1b7a6d",
    "safetyScore": 0,
    "currentVehicle": "Chevrolet Silverado",
    "vehicleId": "e4a7d4d7-1e8b-4e48-bc45-f5d8b9c7a6e5",
    "status": "Online"
  },
  {
    "name": "Chris Jordan",
    "id": "7d5e3a8b-4a9f-49d1-8c7a-2e6f1b4c9a3d",
    "safetyScore": 0,
    "currentVehicle": "Toyota Camry",
    "vehicleId": "b2c4d4e9-5e2a-4b1f-a3f9-6d4e7b2d1a3f",
    "status": "Online"
  },
  {
    "name": "Emily Chris",
    "id": "6c7a2e8d-5f4b-49a1-9b6f-4e2f3d1a7c5d",
    "safetyScore": 0,
    "currentVehicle": "Nissan Altima",
    "vehicleId": "4e2d7c8e-9a4b-46d1-b9a2-6c8d7a5b4e3d",
    "status": "Offline"
  },
  {
    "name": "Jane Taylor",
    "id": "5b4e3c9d-8a6f-49d1-9b7a-7e2d1c4a5e6d",
    "safetyScore": 0,
    "currentVehicle": "Honda Accord",
    "vehicleId": "c4d9b6e7-98d4-4e32-bf1a-3d8e6e4a5b7d",
    "status": "Offline"
  },
  {
    "name": "John Emily",
    "id": "4a7e2b9d-5c3f-49a1-9b6a-6e2d4c1b8a7d",
    "safetyScore": 0,
    "currentVehicle": "Ford F-150",
    "vehicleId": "2b9d0b6a-7c9f-4d1f-bc3a-6d7e5e9c8b7d",
    "status": "Online"
  },
  {
    "name": "Pat Morgan",
    "id": "3c8e4b5a-9a2f-49d1-9c7a-2e6d1f4a5b3d",
    "safetyScore": 0,
    "currentVehicle": "Chevrolet Silverado",
    "vehicleId": "0c6d4e5a-2a9b-4d1f-98bb-7f4e2d1a3c9b",
    "status": "Online"
  },
  {
    "name": "Casey Chris",
    "id": "2e5a4c9d-7b3f-49a1-9b6a-4e2d1f3a6c7d",
    "safetyScore": 0,
    "currentVehicle": "Toyota Camry",
    "vehicleId": "a2d7f6a5-4e8b-42d4-9b4a-6d8e7c4b5a3d",
    "status": "Offline"
  }]