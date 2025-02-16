"use client";
import React, { useState, useEffect, useRef } from "react";
import { Box, Typography, Divider } from "@mui/material";
import styles from "./page.module.css";
import { LiveFrame, ServerResponse, Event } from "./types";
import Image from "next/image";

export default function BodyDemo() {
  const [livePredictStream, setLivePredictStream] = useState<LiveFrame[]>([]);
  const [events, setEvents] = useState<Event[]>([]);
  const bodyImgRef = useRef(null);

  useEffect(() => {
    const eventSource = new EventSource(
      `http://127.0.0.1:5000/body_stream_view`
    );
    eventSource.onmessage = (event) => {
      const data: ServerResponse = JSON.parse(event.data);

      if (bodyImgRef.current) {
        bodyImgRef.current.src = `data:image/jpg;base64,${data.image}`;
      }

      addLivePrediction({
        value: data.prediction,
        probability: parseFloat(data.probability),
        frameNum: parseInt(data.frame_number),
        timestamp: parseInt(data.timestamp),
        processing_time: data.processing_time,
        camera: "Body",
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
            Body Feed With Real Time Predictions
          </Typography>
          <Box className={styles.cameraWrapper}>
            <Image
              ref={bodyImgRef}
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
          {livePredictStream.map((prediction, index) => (
            <Box key={index} className={styles.liveClasBox}>
              <Typography textAlign="center" width={150}>
                {`Frame #${prediction.frameNum}`}
              </Typography>
              <Divider orientation="vertical" flexItem />
              <Typography textAlign="center" width={150}>
                {`Timestamp ${prediction.timestamp}`}
              </Typography>
              <Divider orientation="vertical" flexItem />
              <Typography textAlign="center" width={150}>
              {`Processing Time ${prediction.processing_time.toFixed(2)}`}
              </Typography>
              <Divider orientation="vertical" flexItem />
              <Typography textAlign="center" width={150}>
                {`Prediction: ${prediction.value}`}
              </Typography>
              <Divider orientation="vertical" flexItem />
              <Typography textAlign="center" width={150}>
                {`Prediction Probability: ${(prediction.probability * 100).toFixed(2)}%`}
              </Typography>
            </Box>
          ))}
        </Box>
      </Box>
    </Box>
  );
}
