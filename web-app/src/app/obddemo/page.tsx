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
} from "@mui/material";
import styles from "./page.module.css";
import { ObdFrame, ObdAggregated } from "./types";

export default function ObdDemo() {
  // State for per-frame OBD data (expects frame_number, etc.)
  const [frameData, setFrameData] = useState<ObdFrame[]>([]);
  // State for per-second aggregated OBD data
  const [aggregatedData, setAggregatedData] = useState<ObdAggregated[]>([]);

  // SSE for per-frame OBD data
  useEffect(() => {
    const frameEventSource = new EventSource(
      "https://ghastly-singular-snake.ngrok.app/obd_per_frame_stream_view"
    );
    frameEventSource.onmessage = (event) => {
      const data: ObdFrame = JSON.parse(event.data);
      // Expecting: speed, rpm, check_engine_on, num_dtc_codes, timestamp, frame_number
      setFrameData((prev) => [data, ...prev].slice(0, 5));
    };
    return () => frameEventSource.close();
  }, []);

  // SSE for per-second aggregated OBD data
  useEffect(() => {
    const aggregatedEventSource = new EventSource(
      "https://ghastly-singular-snake.ngrok.app/obd_per_second_stream_view"
    );
    aggregatedEventSource.onmessage = (event) => {
      const data: ObdAggregated = JSON.parse(event.data);
      // Expecting: speed (avg), rpm (avg), check_engine_on (majority), num_dtc_codes (avg), timestamp
      setAggregatedData((prev) => [data, ...prev].slice(0, 5));
    };
    return () => aggregatedEventSource.close();
  }, []);

  // Helper function to display check_engine_on status
  const renderCheckEngineStatus = (value: boolean | null) => {
    if (value === null) return "Unknown";
    return value ? "Yes" : "No";
  };

  // Helper function to display speed
  const renderSpeed = (speed: number) => {
    return speed === -1 ? "Unknown" : speed;
  };

  // Helper function to display RPM
  const renderRPM = (rpm: number) => {
    return rpm === -1 ? "Unknown" : rpm;
  };

  // Helper function to display number of DTC codes
  const renderDtcCodes = (num: number) => {
    return num === -1 ? "Unknown" : num;
  };

  return (
    <Box className={styles.pageContainer}>
      <Box className={styles.section}>
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
                  <TableCell>{renderSpeed(frame.speed)}</TableCell>
                  <TableCell>{renderRPM(frame.rpm)}</TableCell>
                  <TableCell>
                    {renderCheckEngineStatus(frame.check_engine_on)}
                  </TableCell>
                  <TableCell>{renderDtcCodes(frame.num_dtc_codes)}</TableCell>
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
                  <TableCell>{renderSpeed(agg.speed)}</TableCell>
                  <TableCell>{renderRPM(agg.rpm)}</TableCell>
                  <TableCell>
                    {renderCheckEngineStatus(agg.check_engine_on)}
                  </TableCell>
                  <TableCell>{renderDtcCodes(agg.num_dtc_codes)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    </Box>
  );
}
