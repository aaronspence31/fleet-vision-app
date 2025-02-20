// types.ts

// Type for per-frame body data coming from the server
export interface BodyServerResponse {
    image: string;
    prediction: string;
    probability: number;
    frame_number: number;
    timestamp: number;
    processing_time: number;
  }
  
  // Type for per-second aggregated body classification data
  export interface AggregatedBodyClassification {
    prediction: string;
    timestamp: number;
  }
  