// types.ts

// Type for per-frame data coming from the server
export interface ServerResponse {
    image: string;
    eye_prediction: string;
    mouth_prediction: string;
    ear_score: number;
    mar_score: number;
    frame_number: number;
    timestamp: number;
    processing_time: number;
  }
  
  // Type for per-second aggregated face classification data
  export interface AggregatedFaceClassification {
    timestamp: number;
    eye_prediction: string;
    mouth_prediction: string;
  }
  