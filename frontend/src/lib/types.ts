export type UserObject = {
    uid: string;
    firstName: string;
    lastName: string;
    email: string;
    pfp: string;
    cid: string;
    eid: string;
    auditLog: {action: string; where: string; timestamp: Date; }[];
}

export type CompanyObject = {
    displayName: string;
    supportEmail: string;
    cid: string;
    employees: string[];
    clientSecret: string;
    clientKey: string;
}

export const defaultUserObject: UserObject = {
    uid: "",
    firstName: "",
    lastName: "",
    email: "",
    eid: "",
    pfp: "",
    cid: "",
    auditLog: []
}

export const defaultCompanyObject: CompanyObject = {
    displayName: "",
    supportEmail: "",
    cid: "",
    clientSecret: "",
    clientKey: "",
    employees: []
}
