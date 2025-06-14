import axios from 'axios'

const http = axios.create({
    baseURL: 'http://127.0.0.1:8000',
    headers: {
        'Content-Type': 'application/json',
    },
    timeout: 10000,
})

export default http