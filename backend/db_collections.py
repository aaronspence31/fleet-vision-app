from pymongo import MongoClient

# add client string
db = MongoClient(
    "mongodb+srv://isaiah:Nadder3415@fleetvision.tmugdte.mongodb.net/?retryWrites=true&w=majority&appName=FleetVision"
)

# add db name
CapstoneDB = db.get_database("fleet-vision")

obd_col = CapstoneDB.get_collection("obd")
model_col = CapstoneDB.get_collection("model")
