from firebase_admin import credentials, firestore, initialize_app

cred = credentials.Certificate("key.json")
default_app = initialize_app(cred)
db = firestore.client()
body_drive_sessions = db.collection("body_drive_sessions")
face_drive_sessions = db.collection("face_drive_sessions")
obd_drive_sessions = db.collection("obd_drive_sessions")
