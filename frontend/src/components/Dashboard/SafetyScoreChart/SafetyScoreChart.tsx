import React, { useState } from "react";
import styles from "./SafetyScoreChart.module.css";
import { PieChart } from "react-minimal-pie-chart";
import { safetyDataObj, initSafetyData } from "./types";
import { Typography, Box } from "@mui/material";

export default function SafetyScoreChart() {
  const score = 75.5;

  const [safetyData, setSafetyData] = useState<safetyDataObj[]>(initSafetyData);

  return (
    <Box className={styles.pageContainer}>
      <Typography variant="h6" pb={2}>
        Driver Safety Score Distribution
      </Typography>
      <Box className={styles.chartContainer} pb={2}>
        <PieChart
          style={{
            fontFamily:
              '"Nunito Sans", -apple-system, Helvetica, Arial, sans-serif',
            fontSize: "8px",
          }}
          data={safetyData}
          radius={50}
          lineWidth={35}
          label={({ dataEntry }) => Math.round(dataEntry.percentage) + "%"}
          labelPosition={81}
          labelStyle={{
            fill: "#fff",
            opacity: 0.75,
            pointerEvents: "none",
          }}
        />
        <Box className={styles.chartLabel}>
          Average Score
          <span className={styles.chartLabelScore}>{score}</span>
        </Box>
      </Box>
      <Box className={styles.legendContainer}>
        {safetyData.map((safetyObj, index) => {
          return (
            <Box className={styles.legendItemContainer} key={index}>
              <Box
                sx={{
                  backgroundColor: safetyObj.color,
                  height: 15,
                  width: 15,
                  borderRadius: "100%",
                }}
              />
              <Typography className={styles.legendItemTitle}>
                {safetyObj.title}
              </Typography>
            </Box>
          );
        })}
      </Box>
    </Box>
  );
}
