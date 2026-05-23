"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { useState } from "react";

export function QueryProvider({ children }: { children: React.ReactNode }) {
  // Instantiate per-component to avoid shared state across requests (Next.js SSR safety)
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 30 * 1000,        // 30s — RAG responses don't change mid-session
            retry: 1,                     // One retry max — avoid hammering slow AI endpoints
            refetchOnWindowFocus: false,  // Chat state shouldn't reset on tab switch
          },
          mutations: {
            retry: 0, // Never auto-retry mutations (uploads, asks) — user should trigger explicitly
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {process.env.NODE_ENV === "development" && (
        <ReactQueryDevtools initialIsOpen={false} buttonPosition="bottom-left" />
      )}
    </QueryClientProvider>
  );
}