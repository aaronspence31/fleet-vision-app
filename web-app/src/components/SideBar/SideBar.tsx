import React from "react";
import styles from "./SideBar.module.css";
import { Box, Typography } from "@mui/material";
import Image from "next/image";
import { Anonymous_Pro } from "next/font/google";
import {
  Person,
  ExitToApp,
  Dashboard as DashboardIcon,
  TimeToLeave,
  Face,
  Accessibility,
  Speed,
} from "@mui/icons-material";
import { useRouter, usePathname } from "next/navigation";
const anonymous = Anonymous_Pro({ weight: "700", subsets: ["latin"] });

const SideBar = () => {
  const router = useRouter();
  const pathName = usePathname();

  return (
    <Box className={styles.sideBar}>
      <Box className={styles.FlexBoxRow} sx={{ color: "#C3D4C6" }}>
        <Image
          src="/images/fleet-vision-logo.png"
          width={90}
          height={80}
          alt="fleet vision logo"
        />
        <Typography
          className={anonymous.className}
          sx={{ typography: { xl: "h4", lg: "h5" } }}
        >
          Fleet Vision
        </Typography>
      </Box>
      <Box style={{ display: "flex", flexDirection: "column", gap: 15 }}>
        <Box
          className={[
            styles.iconContainer,
            pathName == "/" && styles.selectedPage,
          ].join(" ")}
          onClick={() => router.push("/")}
        >
          <DashboardIcon className={styles.icon} />
          <Typography
            className={anonymous.className}
            sx={{ typography: { xl: "h5", lg: "h6" } }}
          >
            Home
          </Typography>
        </Box>
        <Box
          className={[
            styles.iconContainer,
            pathName == "/vehicleinfo" && styles.selectedPage,
          ].join(" ")}
          onClick={() => router.push("/vehicleinfo")}
        >
          <TimeToLeave className={styles.icon} />
          <Typography
            className={anonymous.className}
            sx={{ typography: { xl: "h5", lg: "h6" } }}
          >
            Vehicle Information
          </Typography>
        </Box>
        <Box
          className={[
            styles.iconContainer,
            pathName == "/facedemo" && styles.selectedPage,
          ].join(" ")}
          onClick={() => router.push("/facedemo")}
        >
          <Face className={styles.icon} />
          <Typography
            className={anonymous.className}
            sx={{ typography: { xl: "h5", lg: "h6" } }}
          >
            Face Feed
          </Typography>
        </Box>
        <Box
          className={[
            styles.iconContainer,
            pathName == "/bodydemo" && styles.selectedPage,
          ].join(" ")}
          onClick={() => router.push("/bodydemo")}
        >
          <Accessibility className={styles.icon} />
          <Typography
            className={anonymous.className}
            sx={{ typography: { xl: "h5", lg: "h6" } }}
          >
            Body Feed
          </Typography>
        </Box>
        <Box
          className={[
            styles.iconContainer,
            pathName == "/obddemo" && styles.selectedPage,
          ].join(" ")}
          onClick={() => router.push("/obddemo")}
        >
          <Speed className={styles.icon} />
          <Typography
            className={anonymous.className}
            sx={{ typography: { xl: "h5", lg: "h6" } }}
          >
            OBD Feed
          </Typography>
        </Box>
      </Box>
      <Box style={{ display: "flex", flexDirection: "column", gap: 15 }}>
        <Box className={styles.iconContainer}>
          <Person className={styles.icon} />
          <Typography
            className={anonymous.className}
            sx={{ typography: { xl: "h5", lg: "h6" } }}
          >
            Profile
          </Typography>
        </Box>
        <Box className={styles.iconContainer}>
          <ExitToApp className={styles.icon} style={{ padding: 15 }} />

          <Typography
            className={anonymous.className}
            sx={{ typography: { xl: "h5", lg: "h6" } }}
          >
            Logout
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};

export default SideBar;
