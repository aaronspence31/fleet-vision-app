"use client";
import React, { useState, useEffect, useRef } from "react";
import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from "@mui/material";
import Image from "next/image";
import styles from "./page.module.css";
import {
  BodyServerResponse,
  AggregatedBodyClassification,
} from "./types";

export default function BodyDemo() {
  // Ref for the live per-frame image stream
  const bodyImgRef = useRef<HTMLImageElement>(null);
  // State for per-second aggregated body classifications
  const [aggregatedData, setAggregatedData] = useState<AggregatedBodyClassification[]>([]);

  // SSE for per-frame image (live stream with drawn predictions)
  useEffect(() => {
    const frameEventSource = new EventSource(
      "https://ghastly-singular-snake.ngrok.app/body_per_frame_stream_view"
    );
    frameEventSource.onmessage = (event) => {
      const data: BodyServerResponse = JSON.parse(event.data);
      // Update the image source using the base64 encoded image data
      if (bodyImgRef.current) {
        bodyImgRef.current.src = `data:image/jpg;base64,${data.image}`;
      }
    };
    return () => frameEventSource.close();
  }, []);

  // SSE for per-second aggregated body classifications
  useEffect(() => {
    const aggregatedEventSource = new EventSource(
      "https://ghastly-singular-snake.ngrok.app/body_per_second_stream_view"
    );
    aggregatedEventSource.onmessage = (event) => {
      const data: AggregatedBodyClassification = JSON.parse(event.data);
      setAggregatedData((prev) => [data, ...prev].slice(0, 3));
    };
    return () => aggregatedEventSource.close();
  }, []);

  // Helper function to render prediction and convert empty/blank predictions to "Unknown"
  const renderPrediction = (prediction: string) => {
    return prediction.trim() === "" ? "Unknown" : prediction;
  };

  return (
    <Box className={styles.pageContainer}>
      {/* Per-frame image section */}
      <Box className={styles.section}>
        <Typography variant="h6" textAlign="center" gutterBottom>
          Body Feed With Real Time Frame Predictions
        </Typography>
        <Box className={styles.cameraContainer}>
          <Box className={styles.cameraWrapper}>
            <Image
              ref={bodyImgRef}
              src=""
              className={styles.video}
              width={1280}
              height={720}
              alt="Live Body Stream"
            />
          </Box>
        </Box>
      </Box>

      {/* Per-second aggregated classification table */}
      <Box className={styles.section}>
        <Typography variant="h5" textAlign="center" gutterBottom>
          Body Data – Per Second Aggregated Classification
        </Typography>
        <TableContainer component={Paper} className={styles.tableContainer}>
          <Table stickyHeader>
            <TableHead className={styles.tableHeader}>
              <TableRow>
                <TableCell>Timestamp</TableCell>
                <TableCell>Prediction</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {aggregatedData.map((agg, index) => (
                <TableRow key={index}>
                  <TableCell>{agg.timestamp}</TableCell>
                  <TableCell>{renderPrediction(agg.prediction)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    </Box>
  );
}
