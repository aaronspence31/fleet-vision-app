export interface FrameBatchType {
  frame_count: number;
  frames: string[];
  session_id: string;
  timestamp_start: number;
  timestamp_end: number;
}

const safetyScoreMap = {
  texting: { scoreModifier: -3 },
  phone_call: { scoreModifier: -1.1 },
  radio: { scoreModifier: -1 },
  reach_Side: { scoreModifier: -4.5 },
  hair_and_makeup: { scoreModifier: -1.8 },
  talking_to_passenger: { scoreModifier: -15 },
  drowsy: { scoreModifier: -5 },
  safe: { scoreModifier: 2 },
};

export const nameToColor = (name: string): string => {
  let hash = 0;
  let i;

  /* eslint-disable no-bitwise */
  for (i = 0; i < name.length; i += 1) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash);
  }

  let color = "#";

  for (i = 0; i < 3; i += 1) {
    const value = (hash >> (i * 8)) & 0xff;
    color += `00${value.toString(16)}`.slice(-2);
  }
  /* eslint-enable no-bitwise */

  return color;
};

export const groupBySessionId = (sessions: FrameBatchType[]) => {
  const grouped = sessions.reduce((acc: any, session: any) => {
    const { session_id } = session;
    if (!acc[session_id]) {
      acc[session_id] = [];
    }
    acc[session_id].push(session);
    return acc;
  }, {});

  return Object.values(grouped);
};

export const getDateTimeValues = (unixTimestamp: number) => {
  const date = new Date(unixTimestamp * 1000);
  
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

const getSafetyScore = (frameBatch: FrameBatchType) => {
  let score = 0;
  frameBatch.frames.forEach((frame: string) => {
    const frameScore = safetyScoreMap[frame];
    if (frameScore) {
      score += frameScore.scoreModifier;
    }
  });
  return Math.ceil((score / frameBatch.frame_count) * 100);

}

export const getSafetyScoreProgression = (session: FrameBatchType[]) => {
  if (!session) return []
  let safetyScores: number[] = []

  session.forEach((sessionInfo: FrameBatchType) => {
    const currentSafetyScore = getSafetyScore(sessionInfo)
    safetyScores.push(currentSafetyScore)
  })

  return safetyScores
}

const capitalize = (str: string, lower = false) =>
  (lower ? str.toLowerCase() : str).replace(/(?:^|\s|["'([{])+\S/g, match => match.toUpperCase());
;

export const countFrameOccurrences = (dataArray: FrameBatchType[]) => {
  if (!dataArray) return []
  const frameTypes = [
      "texting", "phone_call", "radio", "reach_side", 
      "talking_to_passenger", "drowsy", "hair_and_makeup"
  ];
  
  // Initialize counts for each frame type
  const frameCounts: { [key: string]: number } = frameTypes.reduce((acc: { [key: string]: number }, frame: string) => {
      acc[frame] = 0;
      return acc;
  }, {});
  
  // Loop over the array and count occurrences
  dataArray.forEach(({ frames }) => {
      frames.forEach(frame => {
          if (frameCounts.hasOwnProperty(frame)) {
              frameCounts[frame] += 1;
          }
      });
  });
  
  // Convert the result to the desired format
  return Object.entries(frameCounts)
  .filter(([_, value]) => value > 0)
  .map(([label, value], index) => ({
      id: index,
      value,
      label: capitalize(label.replace("talking_to_passenger", "Talking").replace("hair_and_makeup", "Adjusting Hair").replaceAll("_", " "))
  }));
}

export const getSafetyScoreBySession = (session: FrameBatchType[]) => {
  if (!session) return 0
  let safetyScore = 0;
  session.forEach((frameBatch: FrameBatchType) => {
    const currSafetyScore = getSafetyScore(frameBatch);
    // console.log("Current Safety Score: ", currSafetyScore);
    safetyScore += currSafetyScore;
  });
  safetyScore = safetyScore / session.length;
  return Math.ceil(safetyScore);
}