"use client";
import { Inter } from "next/font/google";
import "./globals.css";
import ParentProvider from "./ParentProvider";
import SideBar from "@/components/SideBar/SideBar";

const inter = Inter({ subsets: ["latin"] });

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <ParentProvider>
        <body
          className={inter.className}
          style={{ display: "flex", flexDirection: "row" }}
        >
          <SideBar />
          <div style={{ width: "80vw" }}>{children}</div>
        </body>
      </ParentProvider>
    </html>
  );
}
