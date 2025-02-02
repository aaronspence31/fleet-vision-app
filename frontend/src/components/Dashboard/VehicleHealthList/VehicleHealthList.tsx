import React, { useState } from "react";
import styles from "./VehicleHealthList.module.css";
import { Typography, Box } from "@mui/material";
import { VehicleHealthListProps } from "./types";

export default function VehicleHealthList({
  vehicleErrors,
  vehicleSelected = false,
}: VehicleHealthListProps) {
  const formatDisplayTime = (timestamp: string): string => {
    const date = new Date(timestamp);
    const month = date.toLocaleString("default", { month: "short" });
    const day = date.getDate();
    let hours = date.getHours();
    const mins = date.getMinutes().toString();
    const displayMins = mins.length < 2 ? "0" + mins : mins;
    return `${month} ${day} ${hours}:${displayMins}`;
  };

  return (
    <Box className={styles.container}>
      <Typography
        variant="h5"
        fontWeight={500}
        fontSize={24}
        textAlign="center"
        mb={2}
      >
        Vehicle Maintenance Alerts
      </Typography>
      <Box className={styles.errorListContainer}>
        {vehicleErrors.map((vehicleInfo, index) => (
          <Box className={styles.ErorrContainer} key={index}>
            <Typography className={styles.errorTime} width={100} height="100%">
              {formatDisplayTime(vehicleInfo.timestamp)}
            </Typography>
            <Typography className={styles.errorDescription}>
              {vehicleInfo.description}
            </Typography>
            <Typography width={60} borderRight="1px solid black" height="100%">
              {vehicleInfo.OBDCode}
            </Typography>
            {!vehicleSelected && (
              <Typography width={30} height="100%">
                {vehicleInfo.vehicleID}
              </Typography>
            )}
          </Box>
        ))}
      </Box>
    </Box>
  );
}
