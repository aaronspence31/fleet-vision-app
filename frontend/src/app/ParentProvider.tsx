"use client";

import React from "react";
import { StyledEngineProvider } from "@mui/system";
import { NotificationProvider } from "@/context/NotificationContext";

export default function ParentProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <StyledEngineProvider injectFirst>
      <NotificationProvider>{children}</NotificationProvider>
    </StyledEngineProvider>
  );
}
