// Wrapper component for the signin page

"use client";
import React, { useState, useEffect } from "react";
import { Box, Typography, Divider } from "@mui/material";
import Image from "next/image";
import { DataGrid, GridToolbarQuickFilter } from "@mui/x-data-grid";
import styles from "./page.module.css";
import { driverData, Driver, driverColumns } from "./types";
import { HealthAndSafetyOutlined, DriveEtaOutlined } from "@mui/icons-material";

function QuickSearchToolbar() {
  return (
    <Box sx={{}} className={styles.searchBar}>
      <GridToolbarQuickFilter
        style={{ flex: 1 }}
        quickFilterParser={(searchInput: string) =>
          searchInput
            .split(",")
            .map((value) => value.trim())
            .filter((value) => value !== "")
        }
        debounceMs={400}
      />
    </Box>
  );
}

function noDriverSelected() {
  return (
    <Box>
      <Typography fontSize={32} textAlign="center">
        No Driver Selected
      </Typography>
      <Image
        src="/images/driverIcon.png"
        alt="Description of the image"
        width={400}
        height={400}
        style={{ opacity: 0.5 }}
      />
      <Typography textAlign="center">
        Select a driver to see a detailed view
      </Typography>
    </Box>
  );
}

interface TempData {
  name: string;
  minEventCount: number;
  maxEventCount: number;
  weightingIndex: number;
}

const recentDriverBehaviourStats: TempData[] = [
  {
    name: "Texting (Right)",
    minEventCount: 30,
    maxEventCount: 120,
    weightingIndex: 0,
  },
  {
    name: "Phone call (Right)",
    minEventCount: 30,
    maxEventCount: 150,
    weightingIndex: 1,
  },
  {
    name: "Texting (Left)",
    minEventCount: 30,
    maxEventCount: 120,
    weightingIndex: 2,
  },
  {
    name: "Phone call (Left)",
    minEventCount: 30,
    maxEventCount: 150,
    weightingIndex: 3,
  },
  { name: "Radio", minEventCount: 30, maxEventCount: 250, weightingIndex: 4 },
  {
    name: "Drinking",
    minEventCount: 30,
    maxEventCount: 300,
    weightingIndex: 5,
  },
  {
    name: "Reach Side",
    minEventCount: 15,
    maxEventCount: 100,
    weightingIndex: 6,
  },
  {
    name: "Hair and Makeup",
    minEventCount: 30,
    maxEventCount: 400,
    weightingIndex: 7,
  },
  {
    name: "Talking to passenger",
    minEventCount: 10,
    maxEventCount: 80,
    weightingIndex: 8,
  },
  {
    name: "Reach Backseat",
    minEventCount: 15,
    maxEventCount: 100,
    weightingIndex: 9,
  },
  {
    name: "Drowsy",
    minEventCount: 30,
    maxEventCount: 170,
    weightingIndex: 10,
  },
  {
    name: "Safe",
    minEventCount: 20000,
    maxEventCount: 30000,
    weightingIndex: 11,
  },
];

const sortedDriverBehaviour: TempData[] = recentDriverBehaviourStats
  .sort((a, b) => b.maxEventCount - a.maxEventCount)
  .slice(1, 5);

const safetyWeightArray = [
  -6.1, -2.2, -6.1, -2.2, -1.9, -1.8, -9.1, -1, -30, -9.1, -10, 1,
];

const generateTempDriverStats = (min: number, max: number): number => {
  return Math.floor(Math.random() * (max - min) + min);
};

const getSafetyScoreDemo = () => {
  let safetyScore = 0;
  let totalFrameCount = 0;
  recentDriverBehaviourStats.forEach((behaviourStat: TempData) => {
    const tempCount = generateTempDriverStats(
      behaviourStat.minEventCount,
      behaviourStat.maxEventCount
    );
    totalFrameCount += tempCount;
    safetyScore += tempCount * safetyWeightArray[behaviourStat.weightingIndex];
  });
  return Math.floor((safetyScore / totalFrameCount) * 100);
};

const analyzeDriverSafety = () => {
  driverData.forEach((driver) => {
    driver.safetyScore = Math.floor(getSafetyScoreDemo());
  });
  return driverData;
};

const displaySafetyScoreColor = (safetyScore: number) => {
  if (safetyScore < 50) {
    return "#ff4a3e";
  } else if (safetyScore >= 80) {
    return "#01aefe";
  }
  return "#f8bb00";
};

export default function Drivers() {
  useEffect(() => {
    setDrivers(analyzeDriverSafety());
  }, []);

  const [selectedDriver, setSelectedDriver] = useState<Driver | null>(null);
  const [drivers, setDrivers] = useState<Driver[] | null>(null);
  const [displaySafetyScore, setDisplaySafetyScore] = useState<boolean>(true);

  const getSafetyRank = (driver: Driver) => {
    if (drivers !== null) {
      const sortedDrivers = [...drivers].sort((a, b) => {
        if (b.safetyScore - a.safetyScore) {
          return b.safetyScore - a.safetyScore;
        }
        return -1;
      });
      return sortedDrivers.indexOf(driver) + 1;
    }
    return 0;
  };

  return (
    <Box className={styles.container}>
      <Box
        className={styles.leftBox}
        sx={{
          "& .header": {
            backgroundColor: "var(--accent-primary-hover)",
            color: "white",
          },
        }}
      >
        <DataGrid
          disableColumnFilter
          disableColumnSelector
          disableDensitySelector
          disableMultipleRowSelection
          disableColumnResize
          disableColumnMenu
          hideFooter
          autosizeOnMount
          columns={driverColumns}
          rows={drivers || []}
          slots={{ toolbar: QuickSearchToolbar }}
          onRowSelectionModelChange={(id) => {
            const selectedRowData = driverData.filter(
              (row) => id[0] == row.id
            )[0];
            setSelectedDriver(selectedRowData);
          }}
          slotProps={{
            toolbar: {
              showquickfilter: true,
            },
          }}
          sx={{
            "& .MuiDataGrid-columnHeader:focus, .MuiDataGrid-cell:focus": {
              outline: "none",
            },
            "& .Mui-selected": {
              backgroundColor: "var(--accent-primary) !important",
              color: "white",
            },
            "& .MuiDataGrid-row:hover": {
              backgroundColor: "rgba(61, 139, 255, 0.5)",
              cursor: "pointer",
            },
          }}
        />
      </Box>
      <Box className={styles.rightBox}>
        {selectedDriver ? (
          <Box
            display={"flex"}
            justifyContent={"center"}
            alignItems={"center"}
            flexDirection={"column"}
          >
            <Typography fontSize={32} textAlign="center">
              {selectedDriver?.name}
            </Typography>
            <Box className={styles.status}>
              <Typography className={styles.statusTitle}>
                {selectedDriver.status}
              </Typography>
              <Box
                className={styles.statusIcon}
                bgcolor={selectedDriver.status == "Online" ? "green" : "red"}
              />
            </Box>
            <Image
              src="/images/personIcon.png"
              alt="profile pic"
              width={300}
              height={300}
              style={{ opacity: 0.5 }}
            />
            <Divider flexItem />
            <Box display="flex" className={styles.infoContainer}>
              <Box className={styles.innerInfoContainer}>
                <DriveEtaOutlined />
                <Typography noWrap>{selectedDriver.currentVehicle}</Typography>
              </Box>
              <Divider orientation="vertical" flexItem />
              <Box className={styles.innerInfoContainer}>
                <HealthAndSafetyOutlined />
                <Typography>Rank {getSafetyRank(selectedDriver)}</Typography>
              </Box>
            </Box>
            <Divider flexItem />
            <div style={{ flexGrow: 2 }}>
              {displaySafetyScore ? (
                <div
                  style={{
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    flexDirection: "column",
                  }}
                >
                  <Box
                    className={styles.behaviourContainer}
                    onClick={() => setDisplaySafetyScore(false)}
                  >
                    <Box className={styles.behaviourTitleBox}>
                      <Typography textAlign="center" fontSize={24}>
                        <Typography
                          className={styles.largeTitle}
                          color={displaySafetyScoreColor(
                            selectedDriver.safetyScore
                          )}
                        >
                          Safety
                        </Typography>
                        Score
                      </Typography>
                    </Box>
                    <Box className={styles.behaviourListContainer}>
                      <Box
                        className={styles.safetyScoreContainer}
                        borderColor={displaySafetyScoreColor(
                          selectedDriver.safetyScore
                        )}
                      >
                        <div className={styles.safetyScore}>
                          {selectedDriver.safetyScore}
                        </div>
                      </Box>
                    </Box>
                  </Box>
                  <Box className={styles.safetyScoreMessage}>
                    Click the safety score for a breakdown
                  </Box>
                </div>
              ) : (
                <div>
                  <Box className={styles.behaviourList}>
                    <Box className={styles.behaviourListTitle}>
                      Most common distraction events
                    </Box>
                    {sortedDriverBehaviour.map((driverBehaviour, index) => (
                      <Box key={index} className={styles.behaviourPill}>
                        <Typography px={1} minWidth={200}>
                          {driverBehaviour.name}
                        </Typography>
                        <Divider orientation="vertical" flexItem />
                        <Typography px={1}>
                          {generateTempDriverStats(
                            driverBehaviour.minEventCount,
                            driverBehaviour.maxEventCount
                          )}
                        </Typography>
                      </Box>
                    ))}
                    <Box
                      className={styles.backButton}
                      onClick={() => setDisplaySafetyScore(true)}
                    >
                      Back to Safety Score
                    </Box>
                  </Box>
                </div>
              )}
            </div>
          </Box>
        ) : (
          noDriverSelected()
        )}
      </Box>
    </Box>
  );
}
