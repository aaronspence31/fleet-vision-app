import { getDocs, query, collection, orderBy, limit, where, writeBatch, doc} from 'firebase/firestore';
import { BodySessionData, FaceSessionData, OBDSessionData } from './general';
import { db } from "@/lib/firebase";

export const loadBodySessions = async () => {
    const session_query = query(collection(db, "body_drive_sessions"), orderBy("created_at", "desc"));
    const sessionsSnapshot = await getDocs(session_query);

    return getBodyData(sessionsSnapshot);
}

export const loadNewBodySessions = async (sessions: BodySessionData[]) => {
    const exclusionList = getCurrentSessionIds(sessions);

    const chunkedIds = [];
    for (let i = 0; i < exclusionList.length; i += 10) {
      chunkedIds.push(exclusionList.slice(i, i + 10));
    }

    const bodySessionsRef = collection(db, "body_drive_sessions");

    const queryPromises = chunkedIds.map(chunk => {
        return getDocs(query(bodySessionsRef, where("session_id", "not-in", chunk)));
      });

    const querySnapshots = await Promise.all(queryPromises);
    const formattedData: BodySessionData[] = [];

    for (const snapshot of querySnapshots) {
        const data = await getBodyData(snapshot);
        formattedData.push(...data);
    }

    return formattedData;
}

const getBodyData = async (sessionSnapshot: any) => {
    const formattedData: BodySessionData[] = [];

    for (const sessionDoc of sessionSnapshot.docs) {
        const BodySessionData = sessionDoc.data();
        const sessionId = sessionDoc.id;

        // Initialize session object
        const sessionObject: any = {
            session_id: sessionId,
            created_at: BodySessionData.created_at.toDate() || 0,
            session_frames: [],
            session_name: BodySessionData.session_name || ""
        };

        // Determine nested collection name
        
        const classificationsRef = query(collection(sessionDoc.ref, "body_drive_session_classifications"), orderBy("timestamp", "desc"));
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


export const loadFaceSessions = async () => {
    const session_query = query(collection(db, "face_drive_sessions"), orderBy("created_at", "desc"));
    const sessionsSnapshot = await getDocs(session_query);

    return getFaceData(sessionsSnapshot);
}

const getFaceData = async (sessionSnapshot: any) => {
    const formattedData: FaceSessionData[] = [];

    for (const sessionDoc of sessionSnapshot.docs) {
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

export const loadNewFaceSessions = async (sessions: FaceSessionData[]) => {
    const exclusionList = getCurrentSessionIds(sessions);

    const chunkedIds = [];
    for (let i = 0; i < exclusionList.length; i += 10) {
      chunkedIds.push(exclusionList.slice(i, i + 10));
    }

    const bodySessionsRef = collection(db, "face_drive_sessions");

    const queryPromises = chunkedIds.map(chunk => {
        return getDocs(query(bodySessionsRef, where("session_id", "not-in", chunk)));
      });

    const querySnapshots = await Promise.all(queryPromises);
    const formattedData: FaceSessionData[] = [];

    for (const snapshot of querySnapshots) {
        const data = await getFaceData(snapshot);
        formattedData.push(...data);
    }

    return formattedData;
}

export const loadOBDSessions = async () => {
    const session_query = query(collection(db, "obd_drive_sessions"), orderBy("created_at", "desc"));
    const sessionsSnapshot = await getDocs(session_query);

    return getOBDData(sessionsSnapshot);
}

const getOBDData = async (sessionSnapshot: any) => {
    const formattedData: OBDSessionData[] = [];

    for (const sessionDoc of sessionSnapshot.docs) {
        const BodySessionData = sessionDoc.data();
        const sessionId = sessionDoc.id;

        // Initialize session object
        const sessionObject: any = {
            session_id: sessionId,
            created_at: BodySessionData.created_at.toDate() || 0,
            session_frames: [],
            session_name: BodySessionData.session_name || ""
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

export const loadNewOBDSessions = async (sessions: OBDSessionData[]) => {
    const exclusionList = getCurrentSessionIds(sessions);

    const chunkedIds = [];
    for (let i = 0; i < exclusionList.length; i += 10) {
      chunkedIds.push(exclusionList.slice(i, i + 10));
    }

    const bodySessionsRef = collection(db, "obd_drive_sessions");

    const queryPromises = chunkedIds.map(chunk => {
        return getDocs(query(bodySessionsRef, where("session_id", "not-in", chunk)));
      });

    const querySnapshots = await Promise.all(queryPromises);
    const formattedData: OBDSessionData[] = [];

    for (const snapshot of querySnapshots) {
        const data = await getOBDData(snapshot);
        formattedData.push(...data);
    }

    return formattedData;
}

const getCurrentSessionIds = (sessions: (BodySessionData[] | FaceSessionData[] | OBDSessionData[])) => {
    return sessions.map((session) => session.session_id);
}