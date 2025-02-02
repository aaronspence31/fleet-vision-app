"use client";

import React, { useState } from "react";
import { Box } from "@mui/material";
import styles from "./navbar.module.css";
import { useRouter, usePathname } from "next/navigation";
import { Notifications, Settings, Person } from "@mui/icons-material";
import { useNotificationContext } from "@/context/NotificationContext";
import { NotificationMenu } from "./NotificationMenu";

export const NavBar = () => {
  const router = useRouter();
  const pathName = usePathname();

  const { count, modalOpen, setModalOpen } = useNotificationContext();

  return (
    <Box className={styles.pageContainer}>
      <Box className={styles.tabContainer}>
        <Box className={styles.navBarTitle}>Fleet Vision</Box>
        <Box
          className={[
            styles.tab,
            pathName == "/dashboard" && styles.active,
          ].join(" ")}
          onClick={() => router.push("/dashboard")}
        >
          Dashboard
        </Box>
        <Box
          className={[styles.tab, pathName == "/drivers" && styles.active].join(
            " "
          )}
          onClick={() => router.push("/drivers")}
        >
          Drivers
        </Box>
        <Box
          className={[
            styles.tab,
            pathName == "/vehicles" && styles.active,
          ].join(" ")}
          onClick={() => router.push("/vehicles")}
        >
          Vehicles
        </Box>
        <Box
          className={[styles.tab, pathName == "/demo" && styles.active].join(
            " "
          )}
          onClick={() => router.push("/demo")}
        >
          Face Demo
        </Box>
        <Box
          className={[
            styles.tab,
            pathName == "/bodydemo" && styles.active,
          ].join(" ")}
          onClick={() => router.push("/bodydemo")}
        >
          Body Demo
        </Box>
      </Box>
      <Box className={styles.iconContainer}>
        <Box position="relative">
          <Notifications className={styles.icon} />
          {/* <Box className={styles.notifCount}>
            <span>{count}</span>
          </Box> */}
        </Box>
        <Settings className={styles.icon} />
        <Person className={styles.icon} />
        <Box>Isaiah Richards</Box>
      </Box>
      {modalOpen && (
        <Box
          position="absolute"
          top={100}
          right={100}
          bgcolor={"red"}
          height={200}
          width={200}
        >
          <NotificationMenu />
          <div>TEST TEST TEST TEST TEST</div>
        </Box>
      )}
    </Box>
  );
};
