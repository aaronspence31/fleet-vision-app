import React from "react";
import { Box, Typography, Divider, Avatar } from "@mui/material";
import styles from "./SafetyScoreList.module.css";
import { SafetyScoreListProps, DisplayDriver } from "./types";
import { nameToColor } from "@/utils/general";

export default function SafetyScoreList(props: SafetyScoreListProps) {
  const { isSafestDrivers, driverDisplayList } = props;

  const displayList: DisplayDriver[] = driverDisplayList
    .sort((a, b) => (isSafestDrivers ? b.score - a.score : a.score - b.score))
    .slice(0, 5);

  return (
    <Box display="flex" alignItems="center" flexDirection="column">
      <Typography variant="h5" fontWeight={500}>
        {isSafestDrivers ? "Safest Drivers" : "Dangerous Drivers"}
      </Typography>
      <Box className={styles.driverListContainer}>
        {displayList.map((driver, index) => {
          return (
            <Box className={styles.driverListItem} key={index}>
              <Box display="flex" flexGrow={2} alignItems="center">
                <Avatar
                  sx={{ bgcolor: nameToColor(driver.name) }}
                  className={styles.avatar}
                >
                  {driver.name.substring(0, 1)}
                </Avatar>
                <Box className={styles.driverListInfoContainer} width={200}>
                  <Typography className={styles.driverListItemText}>
                    {driver.name}
                  </Typography>
                  <Typography className={styles.driverListItemSubtext}>
                    Driver
                  </Typography>
                </Box>
              </Box>
              <Box className={styles.driverListInfoContainer} flexGrow={1}>
                <Typography className={styles.driverListItemText}>
                  {driver.score}
                </Typography>
                <Typography className={styles.driverListItemSubtext}>
                  Safety Score
                </Typography>
              </Box>
              <Box
                className={styles.driverListInfoContainer}
                flexGrow={1}
                width={130}
              >
                <Typography className={styles.driverListItemText}>
                  {driver.currentVehicle}
                </Typography>
                <Typography className={styles.driverListItemSubtext}>
                  Current Vehicle
                </Typography>
              </Box>
            </Box>
          );
        })}
      </Box>
    </Box>
  );
}
