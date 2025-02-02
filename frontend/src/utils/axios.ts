import axios, { AxiosInstance } from 'axios'
const serverURL = 'http://127.0.0.1:5000'

export const getLiveLabels = () => {
    
    return axios.get(`${serverURL}/predictions`)
}

export const getEventLabels = () => {
    return axios.get(`${serverURL}/`)
}
