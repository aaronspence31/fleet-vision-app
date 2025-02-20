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
  Button,
} from "@mui/material";
import Image from "next/image";
import styles from "./page.module.css";
import { BodyServerResponse, AggregatedBodyClassification } from "./types";

export default function BodyDemo() {
  const [driveSessionActive, setDriveSessionActive] = useState(false);
  const [loadingSession, setLoadingSession] = useState(true);

  // Check if a session is active when the page loads
  useEffect(() => {
    async function checkSession() {
      try {
        const res = await fetch("https://ghastly-singular-snake.ngrok.app/is_session_active");
        if (res.ok) {
          setDriveSessionActive(true);
        } else {
          setDriveSessionActive(false);
        }
      } catch (error) {
        console.error("Error checking session", error);
        setDriveSessionActive(false);
      } finally {
        setLoadingSession(false);
      }
    }
    checkSession();
  }, []);

  // Handler to start a drive session
  async function handleStartDriveSession() {
    try {
      const res = await fetch("https://ghastly-singular-snake.ngrok.app/start_drive_session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_name: "default" }),
      });
      if (res.ok) {
        setDriveSessionActive(true);
      } else {
        console.error("Failed to start drive session");
      }
    } catch (error) {
      console.error("Error starting drive session", error);
    }
  }

  // Handler to stop the drive session
  async function handleStopDriveSession() {
    try {
      const res = await fetch("https://ghastly-singular-snake.ngrok.app/stop_drive_session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_name: "default" }),
      });
      if (res.ok) {
        setDriveSessionActive(false);
      } else {
        console.error("Failed to stop drive session");
      }
    } catch (error) {
      console.error("Error stopping drive session", error);
    }
  }

  // References and state for stream data
  const bodyImgRef = useRef<HTMLImageElement>(null);
  const [aggregatedData, setAggregatedData] = useState<AggregatedBodyClassification[]>([]);

  // Start the per-frame stream only when the session is active
  useEffect(() => {
    if (!driveSessionActive) return;
    const frameEventSource = new EventSource(
      "https://ghastly-singular-snake.ngrok.app/body_per_frame_stream_view"
    );
    frameEventSource.onmessage = (event) => {
      const data: BodyServerResponse = JSON.parse(event.data);
      if (bodyImgRef.current) {
        bodyImgRef.current.src = `data:image/jpg;base64,${data.image}`;
      }
    };
    return () => frameEventSource.close();
  }, [driveSessionActive]);

  // Start the aggregated stream only when the session is active
  useEffect(() => {
    if (!driveSessionActive) return;
    const aggregatedEventSource = new EventSource(
      "https://ghastly-singular-snake.ngrok.app/body_per_second_stream_view"
    );
    aggregatedEventSource.onmessage = (event) => {
      const data: AggregatedBodyClassification = JSON.parse(event.data);
      setAggregatedData((prev) => [data, ...prev].slice(0, 3));
    };
    return () => aggregatedEventSource.close();
  }, [driveSessionActive]);

  // Helper to render predictions
  const renderPrediction = (prediction: string) => {
    return prediction.trim() === "" ? "Unknown" : prediction;
  };

  // Render loading state
  if (loadingSession) {
    return (
      <Box className={styles.pageContainer} textAlign="center" mt={4}>
        <Typography variant="h6">Loading...</Typography>
      </Box>
    );
  }

  // Render start button if no session is active
  if (!driveSessionActive) {
    return (
      <Box className={styles.pageContainer} textAlign="center" mt={4}>
        <Button variant="contained" color="success" onClick={handleStartDriveSession}>
          Start Drive Session
        </Button>
      </Box>
    );
  }

  // Render main content if a session is active
  return (
    <Box className={styles.pageContainer}>
      {/* Red Stop Session Button at the top */}
      <Box textAlign="center" mt={2}>
        <Button variant="contained" color="error" onClick={handleStopDriveSession}>
          Stop Drive Session
        </Button>
      </Box>
      {/* Per-frame image section */}
      <Box className={`${styles.section} margin-top-class`}>
        <Typography variant="h6" textAlign="center" gutterBottom>
          Body Feed With Real Time Frame Classifications
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
      {/* Aggregated classification table */}
      <Box className={styles.section}>
        <Typography variant="h5" textAlign="center" gutterBottom>
          Body Data – Per Second Aggregated Classifications
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
