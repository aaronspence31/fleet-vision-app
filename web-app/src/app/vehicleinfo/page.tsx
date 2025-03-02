"use client";
import React, { useState, useEffect } from "react";
import { Box, Typography } from "@mui/material";
import styles from "./page.module.css";
import Image from "next/image";
import SpeedGauge from "@/components/VehicleInfoPage/SpeedGauge";
import RpmGauge from "@/components/VehicleInfoPage/RPMGauge";
import InfoBottom from "@/components/VehicleInfoPage/InfoBottom";
import InfoTop from "@/components/VehicleInfoPage/InfoTop";
import { Anonymous_Pro } from "next/font/google";
import { loadOBDSessions } from "@/utils/firestore";
import Select from "react-select";
import Backdrop from "@mui/material/Backdrop";
import CircularProgress from "@mui/material/CircularProgress";
import {
  OBDSessionData,
  getAverageRPM,
  getAverageSpeed,
  calculateDistance,
  generateSessionSelectList,
} from "@/utils/general";

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

const VehicleInfo = () => {
  const [obdData, setObdData] = useState<OBDSessionData[]>(
    [] as OBDSessionData[]
  );
  const [selectedSessionIndex, setSelectedSessionIndex] = useState<number>(0);
  const [selectionList, setSelectionList] = useState<
    { value: String; label: string }[]
  >([]);

  useEffect(() => {
    const loadData = async () => {
      const localOBDData = await loadOBDSessions();
      setObdData(localOBDData);
    };
    loadData();
  }, []);

  useEffect(() => {
    setSelectionList(generateSessionSelectList(obdData));
  }, [obdData]);

  const handleSelectedSessionUpdate = (e: any) => {
    const index = obdData.findIndex((session) => session.session_id == e.value);
    setSelectedSessionIndex(index);
  };

  return (
    <div>
      <Backdrop
        sx={(theme) => ({ color: "#fff", zIndex: theme.zIndex.drawer + 1 })}
        open={obdData.length == 0}
      >
        <CircularProgress color="inherit" />
      </Backdrop>
      <div
        style={{
          opacity: obdData.length == 0 ? 0 : 1,
        }}
      >
        <Box className={styles.SessionSelectContainer}>
          {selectionList.length !== 0 && (
            <Select
              options={selectionList}
              defaultValue={selectionList[0]}
              styles={reactSelectStyles}
              onChange={handleSelectedSessionUpdate}
            />
          )}
        </Box>
        <Box
          style={{
            position: "relative",
            backgroundColor: "#EEF5F4",
            width: "100%",
            height: "100vh",
          }}
        >
          <Image
            src="/images/map.png"
            width={1500}
            height={400}
            alt="Dashboard centre"
            style={{ marginTop: 250 }}
          />
          <Typography
            sx={{ typography: { xl: "h1", lg: "h2" } }}
            className={anonymous.className}
            style={{
              position: "absolute",
              left: 0,
              right: 0,
              marginInline: "auto",
              top: 300,
              textAlign: "center",
            }}
          >
            Fleet Vision <br /> Car Dashboard
          </Typography>
          <Box style={{ position: "absolute", right: -80, top: 280 }}>
            <SpeedGauge
              value={getAverageSpeed(obdData[selectedSessionIndex])}
            />
          </Box>
          <Box style={{ position: "absolute", left: 0, top: 280 }}>
            <RpmGauge value={getAverageRPM(obdData[selectedSessionIndex])} />
          </Box>
          <Box
            style={{
              position: "absolute",
              left: 0,
              right: 0,
              width: 500,
              marginInline: "auto",
              bottom: 200,
            }}
          >
            <InfoBottom
              tripDistance={calculateDistance(
                obdData[selectedSessionIndex]?.session_frames
              )}
            />
          </Box>
          <Box
            style={{
              position: "absolute",
              left: 0,
              right: 0,
              width: 800,
              marginInline: "auto",
              top: 165,
            }}
          >
            <InfoTop />
          </Box>
        </Box>
      </div>
    </div>
  );
};

export default VehicleInfo;
