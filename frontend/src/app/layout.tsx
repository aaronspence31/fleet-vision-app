"use client";
import { Inter } from "next/font/google";
import "./globals.css";
import { NavBar } from "@/components/Navbar/Navbar";
import ParentProvider from "./ParentProvider";

const inter = Inter({ subsets: ["latin"] });

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <ParentProvider>
        <body className={inter.className}>
          <div>{children}</div>
        </body>
      </ParentProvider>
    </html>
  );
}
