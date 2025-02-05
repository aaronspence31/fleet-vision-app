"use client";
import React, { useState, useEffect } from "react";
import { Box, Typography, Button } from "@mui/material";
import styles from "./page.module.css";
import DisplayCard from "@/components/common/DisplayCard";
import { getData, getDocumentsByRecentSession } from "@/utils/firestore";
import { Anonymous_Pro } from "next/font/google";
import { PieChart } from "@mui/x-charts/PieChart";
import { LineChart } from "@mui/x-charts/LineChart";
import SideBar from "@/components/SideBar/SideBar";
import {
  FrameBatchType,
  getDateTimeValues,
  groupBySessionId,
  countFrameOccurrences,
  getSafetyScoreProgression,
  getSafetyScoreBySession,
} from "@/utils/general";

const anonymous = Anonymous_Pro({ weight: "700", subsets: ["latin"] });

function SessionPill(session: FrameBatchType[] | null) {
  if (!session) {
    return;
  }
  const sortedSession = [...session].sort(
    (a, b) => a.timestamp_start - b.timestamp_start
  );

  let safetyScore = getSafetyScoreBySession(session);

  const { date: startDate, time: startTime } = getDateTimeValues(
    sortedSession[0].timestamp_start
  );
  const { date: endDate, time: endTime } = getDateTimeValues(
    sortedSession[sortedSession.length - 1].timestamp_end
  );
  return (
    <Box className={styles.sessionPill}>
      <Typography
        sx={{ typography: { xl: "body1", lg: "body2" } }}
        style={{ color: "#3d8cff" }}
        className={anonymous.className}
      >
        Safety Score: {safetyScore}
      </Typography>
      <Typography
        className={anonymous.className}
        sx={{ typography: { xl: "body1", lg: "body2" } }}
      >
        {`${startDate} ${startTime}   →   ${endDate} ${endTime}`}
      </Typography>
    </Box>
  );
}

export default function Dashboard() {
  const [bodyData, setBodyData] = useState<FrameBatchType[] | null>(null);
  const [faceData, setFaceData] = useState<FrameBatchType[] | null>(null);
  const [recentSession, setRecentSession] = useState<FrameBatchType[] | null>(
    null
  );

  useEffect(() => {
    console.log("Loading Data");
    console.log("Width: ", window.innerWidth);
    const loadData = async () => {
      const localBodyData: FrameBatchType[] = await getData(
        "body_drive_sessions"
      );
      setBodyData(localBodyData);
      const localFaceData = await getData("face_drive_sessions");
      setFaceData(localFaceData);
      const recentSession = await getDocumentsByRecentSession(
        "body_drive_sessions"
      );
      setRecentSession(recentSession);
    };
    loadData();
  }, []);

  return (
    <Box className={styles.pageContainer}>
      <Box className={styles.pageContent}>
        <Box className={styles.pageHeaderBar}>
          <Typography
            sx={{ typography: { xl: "h3", lg: "h4" } }}
            className={anonymous.className}
          >
            Dashboard
          </Typography>
          <Box className={styles.safetyScoreDisplay}>
            <Typography
              sx={{ typography: { xl: "h3", lg: "h4" } }}
              className={anonymous.className}
              style={{ lineHeight: "2.5rem", marginRight: "2rem" }}
            >
              <span>Safety</span>
              <br />
              Score
            </Typography>
            <Typography
              sx={{ typography: { xl: "h1", lg: "h2" } }}
              className={anonymous.className}
              style={{ color: "#01b5e1" }}
            >
              {getSafetyScoreBySession(
                recentSession || ([] as FrameBatchType[])
              )}
            </Typography>
          </Box>
        </Box>
        <Box className={styles.pageContentBody}>
          <DisplayCard>
            <Box className={styles.safetyScoreOverSessionContainer}>
              <Typography
                className={`${anonymous.className} ${styles.DistractionBreakdownTitle}`}
                sx={{ typography: { xl: "h3", lg: "h4" } }}
              >
                Safety Score Progression
              </Typography>
              <LineChart
                bottomAxis={null}
                series={[
                  {
                    data: getSafetyScoreProgression(
                      recentSession || ([] as FrameBatchType[])
                    ),
                  },
                ]}
                width={500}
                height={250}
                disableAxisListener
              />
              <Typography
                className={`${anonymous.className} ${styles.DistractionBreakDownMessage}`}
                sx={{ typography: { xl: "body1", lg: "body2" } }}
              >
                Each point represents the safety score during the{" "}
                <span style={{ fontStyle: "italic" }}>Xth</span> minute of
                driving
              </Typography>
            </Box>
          </DisplayCard>
          <DisplayCard>
            <Typography
              className={`${anonymous.className} ${styles.DistractionBreakdownTitle}`}
              sx={{ typography: { xl: "h3", lg: "h4" } }}
            >
              OBD <br />2<br />
              Stuff
            </Typography>
          </DisplayCard>
          <DisplayCard>
            <Box
              style={{
                width: "100%",
                display: "flex",
                alignItems: "center",
                flexDirection: "column",
                justifyContent: "space-between",
              }}
            >
              <Typography
                className={`${anonymous.className} ${styles.DistractionBreakdownTitle}`}
                sx={{ typography: { xl: "h3", lg: "h4" } }}
              >
                Recent Session Scores
              </Typography>
              <Box className={styles.sessionsContainer}>
                {(
                  groupBySessionId(
                    bodyData || ([] as FrameBatchType[])
                  ) as FrameBatchType[][]
                )
                  .slice(0, 3)
                  .map((session) => {
                    return SessionPill(session);
                  })}
              </Box>
              <Button className={styles.viewSessionsButton}>View All</Button>
            </Box>
          </DisplayCard>
          <DisplayCard>
            <Box
              className={styles.FlexBoxRow}
              style={{ width: "100%", height: "100%" }}
            >
              <Typography
                className={`${anonymous.className} ${styles.DistractionBreakdownTitle}`}
                sx={{ typography: { xl: "h3", lg: "h4" } }}
              >
                Distraction <br /> Event <br /> Breakdown
                <br />
                <span className={styles.DistractionBreakdownDisclaimer}>
                  Most Recent Session
                </span>
              </Typography>

              <Box
                style={{
                  minWidth: "60%",
                  minHeight: "100%",
                  width: "60%",
                  height: "100%",
                  maxWidth: "60%",
                  maxHeight: "100%",
                }}
              >
                <PieChart
                  series={[
                    {
                      data: countFrameOccurrences(
                        bodyData || ([] as FrameBatchType[])
                      ),
                      innerRadius: 30,
                      outerRadius: 100,
                      paddingAngle: 5,
                      cornerRadius: 5,
                      cy: "35%",
                      highlightScope: { fade: "global", highlight: "item" },
                      faded: {
                        innerRadius: 30,
                        additionalRadius: -30,
                        color: "gray",
                      },
                    },
                  ]}
                  sx={{
                    "& .MuiChartsLegend-series text": {
                      fontSize: "0.7em !important",
                    },
                  }}
                  slotProps={{
                    legend: {
                      direction: "row",
                      position: { vertical: "bottom", horizontal: "middle" },
                    },
                  }}
                />
              </Box>
            </Box>
          </DisplayCard>
        </Box>
      </Box>
    </Box>
  );
}
