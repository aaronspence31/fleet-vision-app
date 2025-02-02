import React, { useState } from "react";
import { Box, Typography } from "@mui/material";
import styles from "./navbar.module.css";
import { useRouter, usePathname } from "next/navigation";
import { Notifications, Settings, Person } from "@mui/icons-material";
import { useNotificationContext } from "@/context/NotificationContext";

export const NotificationMenu = () => {
  const router = useRouter();
  const pathName = usePathname();

  const { notifications, addNotification, deleteNotification, count } =
    useNotificationContext();

  const notifColour = (notifType: "error" | "warn" | "info") => {
    if (notifType == "error") return "red";
    else if (notifType == "warn") return "yellow";
    return "green";
  };

  return (
    <Box className={styles.notifMenu}>
      {count ? (
        <Box>
          {notifications.map((notif) => (
            <Box bgcolor={notifColour(notif.type)}>
              <Typography>{notif.message}</Typography>
              <Typography>{notif.originPage}</Typography>
            </Box>
          ))}
        </Box>
      ) : (
        <Box>No Notifications</Box>
      )}
    </Box>
  );
};
