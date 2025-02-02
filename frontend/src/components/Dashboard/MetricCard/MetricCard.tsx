import React, { useState } from "react";
import styles from "./MetricCard.module.css";
import { Typography, Box } from "@mui/material";
import { MetricCardProps } from "./types";

export default function MetricCard(props: MetricCardProps) {
  const { title, value, subtext, extraInfo } = props;

  return (
    <Box className={styles.container}>
      <Typography className={styles.title}>{title}</Typography>
      <Typography className={styles.value}>{value}</Typography>
      <Typography className={styles.subtext}>{subtext}</Typography>
      <Typography className={styles.extraInfo}>({extraInfo})</Typography>
    </Box>
  );
}
