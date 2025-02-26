export interface BodySessionData {
  created_at: Date
  session_id: string
  session_frames: BodySessionDataFrames[]
  session_name?: string
}

interface BodySessionDataFrames { classification: string, timestamp: number }

export interface FaceSessionData {
  created_at: Date
  session_id: string
  session_frames: FaceSessionDataFrames[]
}

interface FaceSessionDataFrames { eyeClassification: string, mouthClassification: string, timestamp: number }

export interface OBDSessionData {
  created_at: Date
  session_id: string
  session_frames: OBDSessionDataFrames[]
}

interface OBDSessionDataFrames { speed: number, rpm: number, checkEngineOn: boolean, numDtcCodes: number, dtcCodes: any[], warningLights: any, timestamp: number }

const safetyScoreMap: any = {
  texting_or_using_phone: -3,
  drinking_or_eating: -1.5,
  talking_on_phone: -1.1,
  reaching_beside_or_behind: -4.5,
  adjusting_hair_or_makeup: -1.8,
  talking_to_passenger: -7,
  Yawning: -4,
  driving_safely: 3,
};

export const getDateTimeValues = (date: Date) => {
  const options: Intl.DateTimeFormatOptions = { 
      month: 'numeric',    
      day: 'numeric',    
      year: '2-digit',   
      timeZone: 'EST'
  };
  
  const formattedDate = date.toLocaleString('en-US', options).replace(/,/g, '');
  
  const timeOptions: Intl.DateTimeFormatOptions = {
      hour: '2-digit',
      minute: '2-digit',
      hour12: true,
      timeZone: 'EST'
  };
  let formattedTime = date.toLocaleString('en-US', timeOptions);
  formattedTime = formattedTime[0] === '0' ? formattedTime.substring(1) : formattedTime
  return {date: formattedDate, time: formattedTime};
}

export const getSafetyScore = (body_session: BodySessionData[], face_session: FaceSessionData[], obd_session: OBDSessionData[]) => {
  let score = 0;
  let scoreCount = 0;
  if (body_session.length == 0 || face_session.length == 0 || obd_session.length == 0) return score
  body_session.forEach((session: BodySessionData, index: number) => {
    let currentStartIndex = 0;
    while (currentStartIndex <= session.session_frames.length) {
      const classifications: any = countFrameOccurrences(session.session_frames.slice(currentStartIndex, currentStartIndex+20))
      currentStartIndex += 20;
      let modeClassification = Object.keys(classifications).reduce((a, b) => classifications[a] > classifications[b] ? a : b);
      modeClassification = modeClassification.toLowerCase().replaceAll(" ", "_");
      if (modeClassification != 'driving_safely'){
        const faceModifier = getFaceModifiers(face_session[index].session_frames.slice(currentStartIndex, currentStartIndex+20));
        const obdModifier = getOBDModifier(obd_session[index].session_frames.slice(currentStartIndex, currentStartIndex+20));
        score += (safetyScoreMap[modeClassification] * faceModifier * obdModifier);
      }
      else{
        score += safetyScoreMap[modeClassification]
      }
      scoreCount++;
  }
  });
  return Math.ceil((score / (scoreCount*safetyScoreMap['driving_safely']))*100);

}

export const getSafetyScoreProgression = (body_session: BodySessionData[], face_session: FaceSessionData[], obd_session: OBDSessionData[]) => {
  if (body_session.length == 0 || face_session.length == 0 || obd_session.length == 0) return []
  let safetyScores: number[] = [0]

  body_session.forEach((session: BodySessionData, index: number) => {
    const currentScore = getSafetyScore([session], [face_session[index]], [obd_session[index]])
    safetyScores.push(currentScore)
    })

  return safetyScores
}


export const countFrameOccurrences = (sessionFrames: any[]) => {
    const classificationCounts: any = {};
  
    // Count occurrences of each classification type
    sessionFrames.forEach(frame => {
        const classification = frame.classification;
        if (classification) {
            classificationCounts[classification] = (classificationCounts[classification] || 0) + 1;
        }
    });

    return classificationCounts
  
}

const getFaceModifiers = (face_session_frames: FaceSessionDataFrames[]) => {
  if (!face_session_frames) return 1;
  const eyeCounts: any = {};
  const mouthCounts: any = {};
  let modifier = 1;

  // Count occurrences of each eyeClassification and mouthClassification
  face_session_frames.forEach((frame: any) => {
      if (frame.eyeClassification) {
          eyeCounts[frame.eyeClassification] = (eyeCounts[frame.eyeClassification] || 0) + 1;
      }
      if (frame.mouthClassification) {
          mouthCounts[frame.mouthClassification] = (mouthCounts[frame.mouthClassification] || 0) + 1;
      }
  });

  const majorityEye = Object.keys(eyeCounts).reduce((a, b) => eyeCounts[a] > eyeCounts[b] ? a : b, '');
  const majorityMouth = Object.keys(mouthCounts).reduce((a, b) => mouthCounts[a] > mouthCounts[b] ? a : b, '');

  if (majorityEye === "Eyes Closed"){
    modifier = modifier * 1.15;
  }
  if (majorityMouth === "Talking"){
    modifier = modifier * 1.05;
  }
  return modifier;

}

const getOBDModifier = (OBD_session_frames: OBDSessionDataFrames[]) => {
  if (!OBD_session_frames) return 1;
  let totalSpeed = 0;
  let speedCount = 0;
  let maxRpm = 0;
  let modifier = 1;

  OBD_session_frames.forEach(frame => {
    // Calculate total speed for average
    if (frame.speed !== undefined && frame.speed !== null) {
        totalSpeed += frame.speed;
        speedCount++;
    }

    // Find max RPM
    if (frame.rpm !== undefined && frame.rpm !== null) {
        maxRpm = Math.max(maxRpm, frame.rpm);
    }
});

  const averageSpeed = speedCount > 0 ? totalSpeed / speedCount : 0;

  if (averageSpeed > 80){
    modifier = modifier * 1.2;
  }
  if (maxRpm > 4500) {
    modifier = modifier * 1.1;
  }

  return modifier;

}

export const getDrowsyPercentange = (face_session: FaceSessionData) => {
  if (!face_session) return 0;
  let eyeClosedCount = 0;

  face_session.session_frames.forEach((frame: FaceSessionDataFrames) => {
      if (frame.eyeClassification === 'Eyes Closed') {
          eyeClosedCount++;
      }
  })
  const eyeClosePercentage = (eyeClosedCount / face_session.session_frames.length) * 100;
  return Math.floor(eyeClosePercentage);
}

export const calculateDistance = (obd_session: OBDSessionDataFrames[]) => {
  if (!Array.isArray(obd_session) || obd_session.length < 2) return 0; 

  obd_session.sort((a, b) => a.timestamp - b.timestamp);

  let totalDistance = 0;

  for (let i = 1; i < obd_session.length; i++) {
    const prev = obd_session[i - 1];
    const curr = obd_session[i];

      const timeDiff = curr.timestamp - prev.timestamp; // Time difference in seconds
      const speedAvg = (prev.speed + curr.speed) / 2; // Average speed (assuming linear change)
      console.log("Speed avg: ", speedAvg)
      console.log("Time diff: ", timeDiff)
      totalDistance += (speedAvg * timeDiff) / 3600; // Convert seconds to hours
  }

  return Math.ceil(totalDistance);
};

export const getAverageSpeed = (obd_session: OBDSessionData) => {
  if (!obd_session) return 0;
  let totalSpeed = 0;
  let speedCount = 0;

  obd_session.session_frames.forEach(frame => {
      if (frame.speed !== undefined && frame.speed !== null) {
          totalSpeed += frame.speed;
          speedCount++;
      }
  });

  return speedCount > 0 ? Math.ceil(totalSpeed / speedCount) : 0;
}

export const getAverageRPM = (obd_session: OBDSessionData) => {
  if (!obd_session) return 0;
  let totalRpm = 0;
  let rpmCount = 0;

  obd_session.session_frames.forEach(frame => {
      if (frame.rpm !== undefined && frame.rpm !== null) {
          totalRpm += frame.rpm;
          rpmCount++;
      }
  });

  return rpmCount > 0 ? totalRpm / rpmCount : 0;
}

export const parseFrameOccurences = (classifications: any) => {
  delete classifications['Driving Safely'];
  return Object.entries(classifications)
  // @ts-ignore
  .filter(([_, value]) => value > 0)
  .map(([label, value], index) => ({
      id: index,
      value,
      label: label
  }));
}

export const generateSessionSelectList = (sessions: BodySessionData[]) => {
  const sessionSelectList: {value: string, label: string}[] = []

  sessions.forEach((session: BodySessionData, index: number) => {
    const sessionSelectObj = {
      value: session.session_id,
      label: session.session_name ? session.session_name : `Session ${index + 1}`
    }
    sessionSelectList.push(sessionSelectObj)
  });

  return sessionSelectList;
}


