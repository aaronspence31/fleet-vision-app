"use client";
import React, { useState, useEffect } from "react";
import { Box, Typography, Button } from "@mui/material";
import styles from "./page.module.css";
import DisplayCard from "@/components/common/DisplayCard";
import { getData, getDocumentsByRecentSession } from "@/utils/firestore";
import { useRouter } from "next/navigation";
import {
  Person,
  ExitToApp,
  Dashboard as DashboardIcon,
  TimeToLeave,
  Face,
  Accessibility,
} from "@mui/icons-material";
import Image from "next/image";
import { Anonymous_Pro } from "next/font/google";
import { PieChart } from "@mui/x-charts/PieChart";
import { LineChart } from "@mui/x-charts/LineChart";
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
        variant="h6"
        style={{ color: "#3d8cff" }}
        className={anonymous.className}
      >
        Safety Score: {safetyScore}
      </Typography>
      <Typography className={anonymous.className}>
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

  const router = useRouter();

  useEffect(() => {
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
      <Box className={styles.sideBar}>
        <Box className={styles.FlexBoxRow} sx={{ color: "#C3D4C6" }}>
          <Image
            src="/images/fleet-vision-logo.png"
            width={90}
            height={80}
            alt="fleet vision logo"
          />
          <Typography className={anonymous.className} variant="h4">
            Fleet Vision
          </Typography>
        </Box>
        <Box style={{ display: "flex", flexDirection: "column", gap: 15 }}>
          <Box className={`${styles.iconContainer} ${styles.selectedPage}`}>
            <DashboardIcon className={styles.icon} />
            <Typography className={anonymous.className} variant="h5">
              Dashboard
            </Typography>
          </Box>
          <Box className={styles.iconContainer}>
            <TimeToLeave className={styles.icon} />
            <Typography className={anonymous.className} variant="h5">
              Vehicle Information
            </Typography>
          </Box>
          <Box
            className={styles.iconContainer}
            onClick={() => router.push("/bodydemo")}
          >
            <Accessibility className={styles.icon} />
            <Typography className={anonymous.className} variant="h5">
              Body Stream
            </Typography>
          </Box>
          <Box
            className={styles.iconContainer}
            onClick={() => router.push("/demo")}
          >
            <Face className={styles.icon} />
            <Typography className={anonymous.className} variant="h5">
              Face Stream
            </Typography>
          </Box>
        </Box>
        <Box style={{ display: "flex", flexDirection: "column", gap: 15 }}>
          <Box className={styles.iconContainer}>
            <Person className={styles.icon} />
            <Typography className={anonymous.className} variant="h5">
              Profile
            </Typography>
          </Box>
          <Box className={styles.iconContainer}>
            <ExitToApp className={styles.icon} style={{ padding: 15 }} />

            <Typography className={anonymous.className} variant="h5">
              Logout
            </Typography>
          </Box>
        </Box>
      </Box>
      <Box className={styles.pageContent}>
        <Box className={styles.pageHeaderBar}>
          <Typography variant="h3" className={anonymous.className}>
            Dashboard
          </Typography>
          <Box className={styles.safetyScoreDisplay}>
            <Typography
              variant="h3"
              className={anonymous.className}
              style={{ lineHeight: "2.5rem", marginRight: "2rem" }}
            >
              <span>Safety</span>
              <br />
              Score
            </Typography>
            <Typography
              variant="h1"
              className={anonymous.className}
              style={{ color: "#01b5e1" }}
            >
              {getSafetyScoreBySession(recentSession)}
            </Typography>
          </Box>
        </Box>
        <Box className={styles.pageContentBody}>
          <DisplayCard>
            <Box className={styles.safetyScoreOverSessionContainer}>
              <Typography
                className={`${anonymous.className} ${styles.DistractionBreakdownTitle}`}
                variant="h3"
              >
                Safety Score Progression
              </Typography>
              <LineChart
                bottomAxis={null}
                series={[
                  {
                    data: getSafetyScoreProgression(recentSession || []),
                  },
                ]}
                width={500}
                height={250}
                disableAxisListener
              />
            </Box>
          </DisplayCard>
          <DisplayCard>
            <Typography
              className={`${anonymous.className} ${styles.DistractionBreakdownTitle}`}
              variant="h3"
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
                variant="h3"
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
            <Box className={styles.FlexBoxRow}>
              <Typography
                className={`${anonymous.className} ${styles.DistractionBreakdownTitle}`}
                variant="h3"
              >
                Distraction <br /> Event <br /> Breakdown
                <br />
                <span className={styles.DistractionBreakdownDisclaimer}>
                  Most Recent Session
                </span>
              </Typography>

              <PieChart
                series={[
                  {
                    data: countFrameOccurrences(bodyData),
                    innerRadius: 30,
                    outerRadius: 100,
                    paddingAngle: 5,
                    cornerRadius: 5,
                    highlightScope: { fade: "global", highlight: "item" },
                    faded: {
                      innerRadius: 30,
                      additionalRadius: -30,
                      color: "gray",
                    },
                    label: (location) => {
                      if (location === "legend") {
                        return "Legend Label";
                      } else if (location === "tooltip") {
                        return "Tooltip Label";
                      }
                      return "Default Label";
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
                    padding: 0,
                  },
                }}
                width={400}
                height={350}
              />
            </Box>
          </DisplayCard>
        </Box>
      </Box>
    </Box>
  );
}
