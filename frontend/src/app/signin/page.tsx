// Wrapper component for the signin page

"use client";
import React, { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import styles from "./page.module.css";
import { useRouter } from "next/navigation";
import FormTextField from "@/components/common/FormItemTextField";
import { SigninInfoForm, SigninInfoFormObject } from "./types";
import Button from "@mui/material/Button";

export default function SignIn() {
  const router = useRouter();

  const [userSigninInfo, setUserSigninInfo] =
    useState<SigninInfoForm>(SigninInfoFormObject);
  const [showPassword, setShowPassword] = useState(false);

  const handlePersonalDetailsFormChange =
    (prop: string) => (event: { target: { value: any } }) => {
      setUserSigninInfo({
        ...userSigninInfo,
        [prop]: event.target.value,
      });
    };

  return (
    <Box className={styles.pageContainer}>
      <Box className={styles.titleContainer}>
        <Typography className={styles.title}>Sign In</Typography>
      </Box>
      <Box sx={{ width: "100%", marginTop: "20px" }}>
        <FormTextField
          labelName="Email"
          value={userSigninInfo.email}
          onChange={handlePersonalDetailsFormChange("email")}
        />{" "}
        {/* Email input field */}
        <br />
        <br />
        <FormTextField
          labelName="Password"
          value={userSigninInfo.password}
          onChange={handlePersonalDetailsFormChange("password")}
          type="password"
          showPassword={showPassword}
          setShowPassword={setShowPassword}
        />{" "}
        {/* Password input field with show/hide password option */}
        <Box
          sx={{
            width: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "flex-start",
            marginTop: "20px",
          }}
        >
          <Typography
            sx={{
              color: "var(--accent-primary)",
              fontSize: "14px",
              fontWeight: "500",
            }}
          >
            Forgot Password?
          </Typography>
        </Box>
        <Button
          variant="contained"
          sx={{
            width: "100%",
            marginTop: "20px",
            borderRadius: "10px",
            padding: "10px",
            textTransform: "none",
            backgroundColor: "white",
            color: "black",
            fontSize: "16px",
            "&:hover": {
              backgroundColor: "white",
              color: "var(--accent-primary)",
            },
            "&:disabled": {
              backgroundColor: "#232323",
              color: "grey",
            },
          }}
          onClick={() => router.push("/dashboard")} // Handle form submission on button click
        >
          Login
        </Button>
      </Box>
    </Box>
  );
}
