/* eslint-disable no-nested-ternary */
/* eslint-disable react/require-default-props */
import {
  FormControl,
  IconButton,
  InputAdornment,
  InputLabel,
  TextField,
} from "@mui/material";
import React from "react";
import { alpha, styled } from "@mui/material/styles";
import InputBase from "@mui/material/InputBase";

interface FormTextFieldProps {
  idName?: string;
  labelName: string;
  value: string;
  onChange: (event: { target: { value: any } }) => void;
  disabled?: boolean;
  type?: string;
  showPassword?: boolean;
  setShowPassword?: any;
  textArea?: boolean;
  darkMode?: boolean;
  multiline?: boolean;
}

const BootstrapInput = styled(InputBase)(({ theme }) => ({
  "label + &": {
    marginTop: theme.spacing(3),
    color: "white",
  },
  "& .MuiInputBase-input": {
    borderRadius: "10px",
    position: "relative",
    backgroundColor: "transparent",
    border: "1px solid",
    borderColor: "#171717",
    fontSize: 16,
    color: "#606060",
    width: "100%",
    padding: "18px 12px",
    transition: theme.transitions.create([
      "border-color",
      "background-color",
      "box-shadow",
    ]),
    // Use the system font instead of the default Roboto font.
    fontFamily: [
      "-apple-system",
      "BlinkMacSystemFont",
      '"Segoe UI"',
      "Roboto",
      '"Helvetica Neue"',
      "Arial",
      "sans-serif",
      '"Apple Color Emoji"',
      '"Segoe UI Emoji"',
      '"Segoe UI Symbol"',
    ].join(","),
    "&:focus": {
      boxShadow: `${alpha(theme.palette.primary.main, 0.25)} 0 0 0 0.2rem`,
      borderColor: "var(--accent-primary)",
    },
  },
}));

export default function FormTextField({
  idName,
  labelName,
  value,
  onChange,
  disabled,
  type,
  showPassword,
  setShowPassword,
  textArea = false,
  darkMode = false,
  multiline = false,
}: FormTextFieldProps) {
  return (
    <FormControl variant="standard" style={{ width: "100%" }}>
      <InputLabel
        shrink
        htmlFor="bootstrap-input"
        style={{ color: "black", fontSize: "18px" }}
      >
        {labelName === "Search" ? (
          ""
        ) : (
          <>
            {labelName}{" "}
            <span style={{ color: "var(--accent-primary)" }}>*</span>
          </>
        )}
      </InputLabel>
      <BootstrapInput
        value={value}
        onChange={onChange}
        id="bootstrap-input"
        placeholder={
          labelName === "First Name"
            ? "John"
            : labelName === "Last Name"
            ? "Doe"
            : labelName === "Work Email"
            ? "hello@company.com"
            : labelName === ("Password" || "Confirm Password")
            ? "pass123"
            : ""
        }
        type={
          type === "password" ? (showPassword ? "text" : "password") : "text"
        }
        disabled={disabled}
        inputComponent={multiline ? "textarea" : "input"}
        sx={{
          "& .MuiInputBase-input": {
            height: multiline ? "120px" : "",
            alignItems: multiline ? "start" : "",
            textAlign: multiline ? "left" : "",
          },
        }}
      />
    </FormControl>
  );
}
