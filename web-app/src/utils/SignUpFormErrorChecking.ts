import { SignUpFormValues } from "@/app/signup/types";

export default async function signUpErrorChecking(formValues: SignUpFormValues) {
  const emailPattern = /^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$/;
  if (formValues.firstName.length === 0) {
    return "First name is required";
  }
  if (formValues.lastName.length === 0) {
    return "Last name is required";
  }
  if (formValues.firstName.length <= 2) {
    return "First name must be longer than 2 characters";
  }
  if (formValues.lastName.length <= 2) {
    return "Last name must be longer than 2 characters";
  }
  if (formValues.email.length === 0) {
    return "Email is required";
  }
  if (emailPattern.test(formValues.email) === false) {
    return "Invalid email address";
  }
  if (formValues.password.length === 0) {
    return "Password is required";
  }
  if (formValues.password.length <= 6) {
    return "Password must be longer than 6 characters";
  }
  return "";
}
