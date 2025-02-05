"use client";

import React from "react";
import { StyledEngineProvider } from "@mui/system";

export default function ParentProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  return <StyledEngineProvider injectFirst>{children}</StyledEngineProvider>;
}
