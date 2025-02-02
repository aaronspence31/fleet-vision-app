interface user {
  UserUUID: string
  Firstname: string
  Lastname: string
  Email: string
  isAdmin: boolean
  Organization: string
}

const testUser: user = {
  "UserUUID": "51f5bf0b-7e8c-43bd-a63e-b6cb546d1385",
  "Firstname": "Isaiah",
  "Lastname": "Richards",
  "Email": "i2richar@uwaterloo.ca",
  "isAdmin": true,
  "Organization": "Fleet Vision Team"
}

interface OBDInfo {
  VehicleUUID: string 
  TroubleCode: string
  CodeType: number
  AffectedSystem: number
  ErrorCode: string
  time: Date
}

const testOBD: OBDInfo = {
  "VehicleUUID": "a1eeac23-8e73-487c-8bba-7ead1fd9adde",
  "TroubleCode": "P",
  "CodeType": 0,
  "AffectedSystem": 4,
  "ErrorCode": "20",
  "time": new Date() //2024-06-27T00:50:47.501Z
}

interface DriverEvent {
  EventClassification: string 
  Time: Date
  Duration: number
  EventUUID: string
  UserUUID: string
  VehicleUUID: string
}

const testDriverEvent: DriverEvent = {
  "EventClassification": "Reach Backseat",
  "Time": new Date(), //2024-06-27T00:50:47.501Z
  "Duration": 7,
  "EventUUID": "49aa5ed8-7f77-44e5-a34c-dff9cd4aae59", 
  "UserUUID": "51f5bf0b-7e8c-43bd-a63e-b6cb546d1385", //from the User interface
  "VehicleUUID": "a1eeac23-8e73-487c-8bba-7ead1fd9adde" //from the OBDInfo interface
}


