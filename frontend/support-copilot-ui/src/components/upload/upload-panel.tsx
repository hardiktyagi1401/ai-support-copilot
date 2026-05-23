"use client";

import { CheckCircle2, RefreshCw } from "lucide-react";
import { useUpload } from "@/lib/hooks/use-upload";
import { UploadZone } from "./upload-zone";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const STATUS_LABELS: Record<string, string> = {
  idle: "",
  uploading: "Uploading file...",
  processing: "Chunking & embedding...",
  success: "Ingestion complete",
  error: "Upload failed",
};

export function UploadPanel() {
  const { upload, isPending, progress, status, result, reset } = useUpload();

  return (
    <div className="flex flex-col gap-6 w-full max-w-xl">
      <div>
        <h2 className="text-lg font-semibold">Document Ingestion</h2>
        <p className="text-sm text-muted-foreground mt-1">
          Upload documents to build your knowledge base. Files are chunked and
          embedded automatically.
        </p>
      </div>

      {/* Upload zone — hidden once successful */}
      {status !== "success" && (
        <UploadZone
          onFileSelect={upload}
          isPending={isPending}
        />
      )}

      {/* Progress bar — visible while uploading or processing */}
      {(status === "uploading" || status === "processing") && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>{STATUS_LABELS[status]}</span>
            <span>{progress}%</span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>
      )}

      {/* Success state */}
      {status === "success" && result && (
        <div className="flex flex-col gap-4 p-5 rounded-xl border border-border bg-card">
          <div className="flex items-center gap-3">
            <CheckCircle2 className="w-5 h-5 text-green-500 shrink-0" />
            <div>
              <p className="text-sm font-medium">{result.filename}</p>
              <p className="text-xs text-muted-foreground mt-0.5">
                Successfully ingested into the knowledge base
              </p>
            </div>
          </div>

          {/* Metadata badges */}
          <div className="flex items-center gap-2 flex-wrap">
            <Badge variant="secondary">
              {result.chunks_created} chunks
            </Badge>
            <Badge
              variant="outline"
              className={cn("text-green-600 border-green-200 bg-green-50")}
            >
              Indexed
            </Badge>
          </div>

          <Button variant="outline" size="sm" onClick={reset} className="w-fit">
            <RefreshCw className="w-4 h-4 mr-2" />
            Upload another
          </Button>
        </div>
      )}

      {/* Error recovery */}
      {status === "error" && (
        <div className="flex items-center justify-between p-4 rounded-lg border border-destructive/30 bg-destructive/5">
          <p className="text-sm text-destructive">Upload failed. Try again.</p>
          <Button variant="ghost" size="sm" onClick={reset}>
            <RefreshCw className="w-4 h-4 mr-2" />
            Retry
          </Button>
        </div>
      )}
    </div>
  );
}