// Wrapper component for the signin page

"use client";
import React, { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import styles from "./page.module.css";
import { useRouter } from "next/navigation";
import FormTextField from "@/components/common/FormItemTextField";
import { SignUpFormValues, initialSignUpFormValues } from "./types";
import Button from "@mui/material/Button";

export default function SignIn() {
  const router = useRouter();

  const [formValues, setFormValues] = useState<SignUpFormValues>(
    initialSignUpFormValues
  );

  const handleChange =
    (prop: string) => (event: { target: { value: any } }) => {
      setFormValues({
        ...formValues,
        [prop]: event.target.value,
      });
    };

  return (
    <Box className={styles.pageContainer}>
      <Box
        className={styles.rightContainer}
        style={{
          marginTop: "50px",
          height: `100%`,
        }}
      >
        <Box className={styles.titleContainer}>
          <Typography className={styles.title}>Sign Up</Typography>
        </Box>
        <Box sx={{ width: "100%", marginTop: "20px" }}>
          <div className={styles.formFieldContainer}>
            <FormTextField
              labelName="Work Email"
              value={formValues.email}
              onChange={handleChange("email")}
            />
          </div>
          <div className={styles.formFieldContainer}>
            <FormTextField
              labelName="First Name"
              value={formValues.firstName}
              onChange={handleChange("firstName")}
            />
          </div>
          <div className={styles.formFieldContainer}>
            <FormTextField
              labelName="Last Name"
              value={formValues.lastName}
              onChange={handleChange("lastName")}
            />
          </div>
          <div className={styles.formFieldContainer}>
            <FormTextField
              labelName="Password"
              value={formValues.password}
              onChange={handleChange("password")}
              type="password"
            />
          </div>
          <Box
            sx={{
              width: "100%",
              display: "flex",
              alignItems: "center",
              justifyContent: "flex-start",
              marginTop: "20px",
            }}
          ></Box>
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
            Sign Up
          </Button>
        </Box>
      </Box>
    </Box>
  );
}
