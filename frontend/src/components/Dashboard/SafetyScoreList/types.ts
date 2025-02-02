export interface SafetyScoreListProps{
    isSafestDrivers: boolean
    driverDisplayList: DisplayDriver[]
}

export interface DisplayDriver {
    name: string
    score: number
    currentVehicle: string
}