// Types for per-frame OBD data
export interface ObdFrame {
    speed: number;
    rpm: number;
    check_engine_on: boolean | null;
    num_dtc_codes: number;
    timestamp: number;
    frame_number: number;
  }
  
  // Types for per-second aggregated OBD data
  export interface ObdAggregated {
    speed: number;
    rpm: number;
    check_engine_on: boolean | null;
    num_dtc_codes: number;
    timestamp: number;
  }
  