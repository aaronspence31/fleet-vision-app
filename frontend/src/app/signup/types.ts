export type SignUpFormValues = {
    firstName: string;
    lastName: string;
    email: string;
    password: string;
};

export const initialSignUpFormValues: SignUpFormValues = {
    firstName: "",
    lastName: "",
    email: "",
    password: "",
};