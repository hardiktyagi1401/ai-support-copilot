import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { apiClient } from "@/lib/api-client";
import { toast } from "sonner";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type UploadStatus =
  | "idle"
  | "uploading"
  | "processing"
  | "success"
  | "error";

export type IngestResponse = {
  document_id: string;
  filename: string;
  status: string;
  message: string;
};

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useUpload() {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<UploadStatus>("idle");
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: async (file: File): Promise<IngestResponse> => {
      const formData = new FormData();
      formData.append("file", file);

      setProgress(0);
      setStatus("uploading");

      const { data } = await apiClient.post<IngestResponse>(
        "/api/v1/ingest/upload",
        formData,
        {
          onUploadProgress: (event) => {
            if (event.total) {
              const pct = Math.round((event.loaded / event.total) * 100);
              // Cap at 90% — the remaining 10% is backend processing time
              setProgress(Math.min(pct, 90));
            }
          },
        }
      );

      // File uploaded — backend is now chunking + embedding
      setStatus("processing");
      setProgress(95);

      return data;
    },

    onSuccess: (data) => {
      setProgress(100);
      setStatus("success");

      // Invalidate any document list queries when they exist in Phase 4
      queryClient.invalidateQueries({ queryKey: ["documents"] });

      toast.success(`"${data.filename}" uploaded`, {
        description: data.message,
      });
    },

    onError: (error: Error) => {
      setStatus("error");
      setProgress(0);
      toast.error("Upload failed", { description: error.message });
    },
  });

  const reset = () => {
    setProgress(0);
    setStatus("idle");
    mutation.reset();
  };

  return {
    upload: mutation.mutate,
    isPending: mutation.isPending,
    progress,
    status,
    result: mutation.data,
    reset,
  };
}