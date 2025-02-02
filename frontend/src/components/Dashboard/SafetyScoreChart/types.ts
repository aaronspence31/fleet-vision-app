
export interface safetyDataObj {
    color: string
    value: number
    title: string
}

export const initSafetyData: safetyDataObj[] = [
    {
        color: '#01aefe',
        value: 20,
        title: "Low Risk"
    },
    {
        color: '#f8bb00',
        value: 20,
        title: "Moderate Risk"
    },
    {
        color: '#ff4a3e',
        value: 20,
        title: "High Risk"
    },
]