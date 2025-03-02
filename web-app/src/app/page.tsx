"use client";
import React, { useState, useEffect } from "react";
import { Box, Typography } from "@mui/material";
import styles from "./page.module.css";
import DisplayCard from "@/components/common/DisplayCard";
import {
  loadBodySessions,
  loadFaceSessions,
  loadOBDSessions,
} from "@/utils/firestore";
import { Anonymous_Pro } from "next/font/google";
import { PieChart } from "@mui/x-charts/PieChart";
import { LineChart } from "@mui/x-charts/LineChart";
import Backdrop from "@mui/material/Backdrop";
import CircularProgress from "@mui/material/CircularProgress";
import Image from "next/image";
import {
  BodySessionData,
  FaceSessionData,
  OBDSessionData,
  getDateTimeValues,
  parseFrameOccurences,
  getSafetyScoreProgression,
  getSafetyScore,
  countFrameOccurrences,
  getDrowsyPercentange,
  generateSessionSelectList,
} from "@/utils/general";
import Select from "react-select";

const anonymous = Anonymous_Pro({ weight: "700", subsets: ["latin"] });
const reactSelectStyles = {
  control: (styles: any) => ({
    ...styles,
    width: "400px",
    height: "50px",
    borderRadius: "10px",
    border: "1px solid #01b5e1",
    "&:hover": {
      borderWidth: "2px",
      transform: "scale(1.02)",
      cursor: "pointer",
    },
  }),
  container: (styles: any) => ({
    ...styles,
    width: "400px",
    height: "50px",
    borderRadius: "10px",
  }),
  singleValue: (styles: any) => ({
    ...styles,
    color: "#01b5e1",
    textAlign: "center",
    fontSize: "1.5rem",
  }),
  option: (styles: any, state: { isSelected: any }) => ({
    ...styles,
    color: "#01b5e1",
    backgroundColor: state.isSelected ? "lightgray" : "white",
    cursor: "pointer",
    "&:hover": {
      backgroundColor: "#3d8cff",
      color: "black",
    },
  }),
  menu: (styles: any) => ({
    ...styles,
    color: "#01b5e1",
    fontSize: "1.2rem",
    cursor: "pointer",
  }),
  menuList: (styles: any) => ({
    ...styles,
    color: "#01b5e1",
    fontSize: "1.2rem",
    cursor: "pointer",
  }),
  dropdownIndicator: (styles: any) => ({
    ...styles,
    display: "none",
  }),
  indicatorSeparator: (styles: any) => ({
    ...styles,
    display: "none",
  }),
};

export default function Dashboard() {
  const [bodyData, setBodyData] = useState<BodySessionData[]>(
    [] as BodySessionData[]
  );
  const [faceData, setFaceData] = useState<FaceSessionData[]>(
    [] as FaceSessionData[]
  );
  const [obdData, setObdData] = useState<OBDSessionData[]>(
    [] as OBDSessionData[]
  );
  const [selectedSessionIndex, setSelectedSessionIndex] = useState<number>(0);

  const [selectionList, setSelectionList] = useState<
    { value: String; label: string }[]
  >([]);

  useEffect(() => {
    const loadData = async () => {
      const localBodyData = await loadBodySessions();
      setBodyData(localBodyData);
      const localFaceData = await loadFaceSessions();
      setFaceData(localFaceData);
      const localOBDData = await loadOBDSessions();
      setObdData(localOBDData);
    };
    loadData();
  }, []);

  useEffect(() => {
    setSelectionList(generateSessionSelectList(bodyData));
    console.log("Body Data: ", bodyData);
  }, [bodyData]);

  const handleSelectedSessionUpdate = (e: any) => {
    const index = bodyData.findIndex(
      (session) => session.session_id == e.value
    );
    setSelectedSessionIndex(index);
  };

  const getSessionPill = (sessionIndex: number) => {
    if (bodyData.length == 0 || faceData.length == 0 || obdData.length == 0)
      return;
    const bodySession = bodyData[sessionIndex];
    const faceSession = faceData[sessionIndex];
    const obdSession = obdData[sessionIndex];
    if (!bodySession || !faceSession || !obdSession) return;

    let safetyScore = getSafetyScore(
      [bodySession],
      [faceSession],
      [obdSession]
    );

    const { date: startDate, time: startTime } = getDateTimeValues(
      bodySession.created_at
    );
    const endTimestampDate = new Date(
      bodySession.session_frames[bodySession.session_frames.length - 1]
        .timestamp * 1000
    );
    const { date: endDate, time: endTime } =
      getDateTimeValues(endTimestampDate);
    return (
      <Box className={styles.sessionPill} key={sessionIndex}>
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
  };

  return (
    <Box className={styles.pageContainer}>
      <Backdrop
        sx={(theme) => ({ color: "#fff", zIndex: theme.zIndex.drawer + 1 })}
        open={
          bodyData.length == 0 || faceData.length == 0 || obdData.length == 0
        }
      >
        <CircularProgress color="inherit" />
      </Backdrop>
      <Box
        className={styles.pageContent}
        sx={{
          opacity:
            bodyData.length == 0 || faceData.length == 0 || obdData.length == 0
              ? 0
              : 1,
        }}
      >
        <Box className={styles.pageHeaderBar}>
          <Typography
            sx={{ typography: { xl: "h3", lg: "h4" } }}
            className={anonymous.className}
          >
            Dashboard
          </Typography>
          <Box>
            <Typography
              sx={{
                typography: { xl: "h5", lg: "h6" },
                textAlign: "center",
                marginBottom: "0.5rem",
              }}
              className={anonymous.className}
            >
              Selected Session
            </Typography>
            {selectionList.length && (
              <Select
                options={selectionList}
                defaultValue={selectionList[0]}
                styles={reactSelectStyles}
                onChange={handleSelectedSessionUpdate}
              />
            )}
          </Box>
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
              {getSafetyScore(
                [bodyData[selectedSessionIndex]],
                [faceData[selectedSessionIndex]],
                [obdData[selectedSessionIndex]]
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
                      bodyData,
                      faceData,
                      obdData
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
                <span style={{ fontStyle: "italic" }}>Nth</span> session
              </Typography>
            </Box>
          </DisplayCard>
          <DisplayCard>
            <Box className={styles.FlexBoxColumn}>
              <Typography
                className={`${anonymous.className} ${styles.DistractionBreakdownTitle}`}
                sx={{ typography: { xl: "h3", lg: "h4" } }}
              >
                Drowsiness Score
              </Typography>
              <Box
                className={styles.ImageMaskContainer}
                style={{
                  background: `linear-gradient(to top, #01b5e1 ${getDrowsyPercentange(
                    faceData[selectedSessionIndex],
                    bodyData[selectedSessionIndex]
                  )}%, transparent ${getDrowsyPercentange(
                    faceData[selectedSessionIndex],
                    bodyData[selectedSessionIndex]
                  )}%)`,
                }}
              >
                <Image
                  src="/images/tired.png"
                  width={230}
                  height={220}
                  alt="Tired Icon"
                  style={{ display: "block" }}
                />
              </Box>
              <Typography
                className={`${anonymous.className} ${styles.DistractionBreakdownTitle}`}
                sx={{ typography: { xl: "b1", lg: "b2" } }}
              >
                The user displayed drowsy tendencies during{" "}
                {getDrowsyPercentange(
                  faceData[selectedSessionIndex],
                  bodyData[selectedSessionIndex]
                )}
                % of the last session
              </Typography>
            </Box>
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
                {(bodyData || [])
                  .slice(selectedSessionIndex, selectedSessionIndex + 3)
                  .map((session, index) => {
                    return getSessionPill(index);
                  })}
              </Box>
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
              </Typography>

              <Box
                style={{
                  minWidth: "60%",
                  minHeight: "100%",
                  width: "60%",
                  height: "350px",
                  maxWidth: "60%",
                  maxHeight: "100%",
                }}
              >
                <PieChart
                  series={[
                    {
                      // @ts-ignore
                      data: parseFrameOccurences(
                        countFrameOccurrences(
                          bodyData.length
                            ? bodyData[selectedSessionIndex].session_frames
                            : []
                        )
                      ),
                      innerRadius: 30,
                      outerRadius: 100,
                      paddingAngle: 5,
                      cornerRadius: 5,
                      cy: "30%",
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
                    "& .MuiChartsLegend-root": {
                      marginTop: "0.5rem !important",
                    },
                  }}
                  slotProps={{
                    legend: {
                      direction: "row", // Keep items in a row-based layout
                      position: { vertical: "bottom", horizontal: "middle" },
                      padding: 10, // Add spacing around the legend
                      itemMarkWidth: 20,
                      itemMarkHeight: 2,
                      markGap: 5,
                      itemGap: 15, // Adjust spacing between legend items for alignment
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
