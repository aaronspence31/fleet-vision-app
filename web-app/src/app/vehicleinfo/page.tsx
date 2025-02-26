"use client";
import React, { useState, useEffect } from "react";
import { Box, Typography } from "@mui/material";
import Image from "next/image";
import SpeedGauge from "@/components/SpeedGauge";
import RpmGauge from "@/components/RPMGauge";
import InfoBottom from "@/components/InfoBottom";
import InfoTop from "@/components/InfoTop";
import { Anonymous_Pro } from "next/font/google";
import { getOBDData } from "@/utils/firestore";
import {
  OBDSessionData,
  getAverageRPM,
  getAverageSpeed,
  calculateDistance,
} from "@/utils/general";

const anonymous = Anonymous_Pro({ weight: "700", subsets: ["latin"] });

const VehicleInfo = () => {
  const [obdData, setObdData] = useState<OBDSessionData[]>(
    [] as OBDSessionData[]
  );

  useEffect(() => {
    const loadData = async () => {
      const localOBDData = await getOBDData();
      setObdData(localOBDData);
    };
    loadData();
  }, []);

  return (
    <div>
      <div>
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
            <SpeedGauge value={getAverageSpeed(obdData[0])} />
          </Box>
          <Box style={{ position: "absolute", left: 0, top: 280 }}>
            <RpmGauge value={getAverageRPM(obdData[0])} />
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
              tripDistance={calculateDistance(obdData[0]?.session_frames)}
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
