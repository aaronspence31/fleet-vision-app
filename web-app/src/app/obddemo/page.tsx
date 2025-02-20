"use client";
import React, { useState, useEffect } from "react";
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
import styles from "./page.module.css";
import { ObdFrame, ObdAggregated } from "./types";

export default function ObdDemo() {
  const [driveSessionActive, setDriveSessionActive] = useState(false);
  const [loadingSession, setLoadingSession] = useState(true);

  // Check session status on page load
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

  // State for per-frame OBD data
  const [frameData, setFrameData] = useState<ObdFrame[]>([]);
  // State for per-second aggregated OBD data
  const [aggregatedData, setAggregatedData] = useState<ObdAggregated[]>([]);

  // Start the per-frame stream when a session is active
  useEffect(() => {
    if (!driveSessionActive) return;
    const frameEventSource = new EventSource(
      "https://ghastly-singular-snake.ngrok.app/obd_per_frame_stream_view"
    );
    frameEventSource.onmessage = (event) => {
      const data: ObdFrame = JSON.parse(event.data);
      setFrameData((prev) => [data, ...prev].slice(0, 10));
    };
    return () => frameEventSource.close();
  }, [driveSessionActive]);

  // Start the aggregated stream when a session is active
  useEffect(() => {
    if (!driveSessionActive) return;
    const aggregatedEventSource = new EventSource(
      "https://ghastly-singular-snake.ngrok.app/obd_per_second_stream_view"
    );
    aggregatedEventSource.onmessage = (event) => {
      const data: ObdAggregated = JSON.parse(event.data);
      setAggregatedData((prev) => [data, ...prev].slice(0, 10));
    };
    return () => aggregatedEventSource.close();
  }, [driveSessionActive]);

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

      <Box className={`${styles.section} ${styles.marginTopClass}`}>
        <Typography variant="h5" textAlign="center" gutterBottom>
          OBD Data – Per Frame
        </Typography>
        <TableContainer component={Paper} className={styles.tableContainer}>
          <Table stickyHeader>
            <TableHead className={styles.tableHeader}>
              <TableRow>
                <TableCell>Timestamp</TableCell>
                <TableCell>Frame #</TableCell>
                <TableCell>Speed</TableCell>
                <TableCell>RPM</TableCell>
                <TableCell>Check Engine On</TableCell>
                <TableCell># DTC Codes</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {frameData.map((frame, index) => (
                <TableRow key={index}>
                  <TableCell>{frame.timestamp}</TableCell>
                  <TableCell>{frame.frame_number}</TableCell>
                  <TableCell>{frame.speed === -1 ? "Unknown" : frame.speed}</TableCell>
                  <TableCell>{frame.rpm === -1 ? "Unknown" : frame.rpm}</TableCell>
                  <TableCell>
                    {frame.check_engine_on === null
                      ? "Unknown"
                      : frame.check_engine_on
                      ? "Yes"
                      : "No"}
                  </TableCell>
                  <TableCell>{frame.num_dtc_codes === -1 ? "Unknown" : frame.num_dtc_codes}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>

      <Box className={styles.section}>
        <Typography variant="h5" textAlign="center" gutterBottom>
          OBD Data – Per Second Aggregated
        </Typography>
        <TableContainer component={Paper} className={styles.tableContainer}>
          <Table stickyHeader>
            <TableHead className={styles.tableHeader}>
              <TableRow>
                <TableCell>Timestamp</TableCell>
                <TableCell>Avg Speed</TableCell>
                <TableCell>Avg RPM</TableCell>
                <TableCell>Check Engine On</TableCell>
                <TableCell>Avg # DTC Codes</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {aggregatedData.map((agg, index) => (
                <TableRow key={index}>
                  <TableCell>{agg.timestamp}</TableCell>
                  <TableCell>{agg.speed === -1 ? "Unknown" : agg.speed}</TableCell>
                  <TableCell>{agg.rpm === -1 ? "Unknown" : agg.rpm}</TableCell>
                  <TableCell>
                    {agg.check_engine_on === null
                      ? "Unknown"
                      : agg.check_engine_on
                      ? "Yes"
                      : "No"}
                  </TableCell>
                  <TableCell>{agg.num_dtc_codes === -1 ? "Unknown" : agg.num_dtc_codes}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    </Box>
  );
}
