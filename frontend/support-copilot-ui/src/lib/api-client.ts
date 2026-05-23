import axios, { AxiosError, AxiosResponse } from "axios";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8001";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30_000,

  headers: {
    "x-tenant-id": "demo-tenant",
  },
});

// ---------------------------------------------------------------------------
// Request interceptor
// ---------------------------------------------------------------------------
apiClient.interceptors.request.use(
  (config) => {
    return config;
  },
  (error: AxiosError) => Promise.reject(error)
);

// ---------------------------------------------------------------------------
// Response interceptor
// ---------------------------------------------------------------------------
apiClient.interceptors.response.use(
  (response: AxiosResponse) => response,

  (error: AxiosError<any>) => {
    const detail = error.response?.data?.detail;

    const message = Array.isArray(detail)
      ? detail.map((d: any) => d.msg).join(", ")
      : detail || error.message || "An unexpected error occurred";

    console.error(
      `[API Error] ${error.response?.status ?? "Network"}: ${message}`
    );

    return Promise.reject(new Error(message));
  }
);