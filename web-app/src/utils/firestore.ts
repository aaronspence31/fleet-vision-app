import { getDocs, query, collection, orderBy, limit } from 'firebase/firestore';
import { BodySessionData, FaceSessionData, OBDSessionData } from './general';
import { db } from "@/lib/firebase";

export const getBodyData = async () => {
    const session_query = query(collection(db, "body_drive_sessions"), orderBy("created_at", "desc"), limit(2));
    const sessionsSnapshot = await getDocs(session_query);

    const formattedData: BodySessionData[] = [];

    for (const sessionDoc of sessionsSnapshot.docs) {
        const BodySessionData = sessionDoc.data();
        const sessionId = sessionDoc.id;

        // Initialize session object
        const sessionObject: any = {
            session_id: sessionId,
            created_at: BodySessionData.created_at.toDate() || 0,
            session_frames: []
        };

        // Determine nested collection name
        const classificationsRef = collection(sessionDoc.ref, "body_drive_session_classifications");
        const classificationsSnapshot = await getDocs(classificationsRef);

        // Populate session_frames array
        classificationsSnapshot.forEach((classificationDoc) => {
            const classificationData = classificationDoc.data();
            sessionObject.session_frames.push({
                classification: classificationData.classification || "",
                timestamp: classificationData.timestamp || 0
            });
        });

        // Add session object to the array
        formattedData.push(sessionObject);
    }

    return formattedData;
}

export const getFaceData = async () => {
    const session_query = query(collection(db, "face_drive_sessions"), orderBy("created_at", "desc"), limit(2));
    const sessionsSnapshot = await getDocs(session_query);

    const formattedData: FaceSessionData[] = [];

    for (const sessionDoc of sessionsSnapshot.docs) {
        const BodySessionData = sessionDoc.data();
        const sessionId = sessionDoc.id;

        // Initialize session object
        const sessionObject: any = {
            session_id: sessionId,
            created_at: BodySessionData.created_at.toDate() || 0,
            session_frames: []
        };

        // Determine nested collection name
        const classificationsRef = collection(sessionDoc.ref, "face_drive_session_classifications");
        const classificationsSnapshot = await getDocs(classificationsRef);

        // Populate session_frames array
        classificationsSnapshot.forEach((classificationDoc) => {
            const classificationData = classificationDoc.data();
            sessionObject.session_frames.push({
                eyeClassification: classificationData.eye_classification || "",
                mouthClassification: classificationData.mouth_classification || "",
                timestamp: classificationData.timestamp || 0
            });
        });

        // Add session object to the array
        formattedData.push(sessionObject);
    }

    return formattedData;
}

export const getOBDData = async () => {
    const session_query = query(collection(db, "obd_drive_sessions"), orderBy("created_at", "desc"), limit(2));
    const sessionsSnapshot = await getDocs(session_query);

    const formattedData: OBDSessionData[] = [];

    for (const sessionDoc of sessionsSnapshot.docs) {
        const BodySessionData = sessionDoc.data();
        const sessionId = sessionDoc.id;

        // Initialize session object
        const sessionObject: any = {
            session_id: sessionId,
            created_at: BodySessionData.created_at.toDate() || 0,
            session_frames: []
        };

        const classificationsRef = collection(sessionDoc.ref, "obd_drive_session_classifications");
        const classificationsSnapshot = await getDocs(classificationsRef);

        // Populate session_frames array
        classificationsSnapshot.forEach((classificationDoc) => {
            const classificationData = classificationDoc.data();
            sessionObject.session_frames.push({
                speed: classificationData.speed,
                rpm: classificationData.rpm,
                checkEngineOn: classificationData.check_engine_on,
                numDtcCodes: classificationData.num_dtc_codes,
                dtcCodes: classificationData.dtc_codes || [],
                warningLights: classificationData.warning_lights || [],
                timestamp: classificationData.timestamp || 0
            });
        });

        // Add session object to the array
        formattedData.push(sessionObject);
    }

    return formattedData;
}
