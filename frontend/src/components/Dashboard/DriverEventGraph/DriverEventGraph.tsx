import React, { useState } from "react";
import styles from "./DriverEventGraph.module.css";
import { Typography, Box } from "@mui/material";
import { BarChart } from "@mui/x-charts/BarChart";

interface TempData {
  name: string;
  count: number;
}

const data: TempData[] = [
  { name: "Texting (Right)", count: 85 },
  { name: "Phone call (Right)", count: 191 },
  { name: "Texting (Left)", count: 66 },
  { name: "Phone call (Left)", count: 77 },
  { name: "Radio", count: 234 },
  { name: "Drinking", count: 142 },
  { name: "Reach Side", count: 94 },
  { name: "Hair and Makeup", count: 187 },
  { name: "Talking to passenger", count: 120 },
  { name: "Reach Backseat", count: 178 },
  { name: "Drowsy", count: 243 },
];

const sortedData: TempData[] = data
  .sort((a, b) => b.count - a.count)
  .slice(0, 7);

export default function DriverEventGraph() {
  const formatLabels = (label: string): string => {
    if (!/\s/.test(label)) return label;
    const labelWords = label.split(" ");
    let masterString = "";
    let currentString = "";
    labelWords.forEach((word) => {
      if (currentString.length + word.length < 11) {
        currentString += word + " ";
      } else {
        masterString += currentString + "\n";
        currentString = word;
      }
    });
    masterString += currentString + "\n";
    return masterString;
  };

  function insertNewlinesEvery10Chars(str: string) {
    const maxLength = 10;
    let result = "";
    let currentLineLength = 0;

    for (let i = 0; i < str.length; i++) {
      result += str[i];
      currentLineLength++;

      // If the current line exceeds maxLength and the next character is a space, add a newline
      if (
        currentLineLength >= maxLength &&
        (str[i] === " " || str[i + 1] === " " || str[i + 1] === undefined)
      ) {
        result += "\n";
        currentLineLength = 0;
      }
    }

    return result;
  }

  return (
    <Box className={styles.container}>
      <Typography
        variant="h5"
        fontWeight={500}
        fontSize={28}
        textAlign="center"
      >
        Most Common Driver Events
      </Typography>
      <Box className={styles.chartContainer}>
        <BarChart
          xAxis={[
            {
              scaleType: "band",
              data: sortedData.map((element: TempData) => element.name),
              valueFormatter: formatLabels,
            },
          ]}
          series={[
            { data: sortedData.map((element: TempData) => element.count) },
          ]}
        />
      </Box>
    </Box>
  );
}
