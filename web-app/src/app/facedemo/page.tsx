"use client";
import React, { useState, useEffect, useRef } from "react";
import { Box, Typography, Divider } from "@mui/material";
import styles from "./page.module.css";
import { LiveFrame, ServerResponse, Event } from "./types";
import Image from "next/image";

export default function FaceDemo() {
  const [livePredictStream, setLivePredictStream] = useState<LiveFrame[]>([]);
  const [events, setEvents] = useState<Event[]>([]);
  const faceImgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    const eventSource = new EventSource(
      `http://ghastly-singular-snake.ngrok.app:5000/face_stream_view`
    );
    eventSource.onmessage = (event) => {
      const data: ServerResponse = JSON.parse(event.data);

      if (faceImgRef.current) {
        faceImgRef.current.src = `data:image/jpg;base64,${data.image}`;
      }

      addLivePrediction({
        eye_prediction: data.eye_prediction,
        mouth_prediction: data.mouth_prediction,
        ear_score: parseFloat(data.ear_score),
        mar_score: parseFloat(data.mar_score),
        frameNum: parseInt(data.frame_number),
        timestamp: parseInt(data.timestamp),
        processing_time: parseFloat(data.processing_time),
        camera: "Face",
      });
    };

    return () => eventSource.close();
  }, []);

  const addLivePrediction = (prediction: LiveFrame) => {
    setLivePredictStream((prev) => [prediction, ...prev.slice(0, 149)]);
  };

  useEffect(() => {
    if (livePredictStream.length > 150) {
      setLivePredictStream((prev) => prev.slice(0, 30));
    }
  }, [livePredictStream]);

  return (
    <Box className={styles.pageContainer}>
      <Box className={styles.cameraContainer}>
        <Box>
          <Typography variant="h6" textAlign="center">
            Face Feed With Real Time Predictions
          </Typography>
          <Box className={styles.cameraWrapper}>
            <Image
              ref={faceImgRef}
              src={""}
              className={styles.video}
              width={1280}
              height={720}
              alt="Live Stream"
            />
          </Box>
        </Box>
      </Box>
      <Box className={styles.listContainer}>
        <Box className={styles.liveClasContainer}>
          <Typography variant="h5" textAlign={"center"} fontWeight={500}>
            Live Frame Classification
          </Typography>
          {livePredictStream
            .filter(prediction => prediction.eye_prediction && prediction.mouth_prediction)
            .map((prediction, index) => (
            <Box key={index} className={styles.liveClasBox}>
              <Typography textAlign="center" width={150}>
                {`Timestamp ${prediction.timestamp}`}
              </Typography>
              <Divider orientation="vertical" flexItem />
              <Typography textAlign="center" width={150}>
                {`Eye Prediction: ${prediction.eye_prediction}`}
              </Typography>
              <Divider orientation="vertical" flexItem />
              <Typography textAlign="center" width={150}>
                {`Mouth Prediction: ${prediction.mouth_prediction}`}
              </Typography>
            </Box>
          ))}
        </Box>
      </Box>
    </Box>
  );
}
