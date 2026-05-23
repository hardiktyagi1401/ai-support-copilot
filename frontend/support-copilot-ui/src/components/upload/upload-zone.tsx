"use client";

import { useCallback, useState } from "react";
import { Upload, FileText, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

type UploadZoneProps = {
  onFileSelect: (file: File) => void;
  isPending: boolean;
  disabled?: boolean;
};

const ACCEPTED_TYPES = [
  "application/pdf",
  "text/plain",
  "text/markdown",
  "application/msword",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
];

const MAX_SIZE_MB = 20;

export function UploadZone({ onFileSelect, isPending, disabled }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  const validate = (file: File): string | null => {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      return "Unsupported file type. Use PDF, TXT, MD, or DOCX.";
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      return `File exceeds ${MAX_SIZE_MB}MB limit.`;
    }
    return null;
  };

  const handleFile = useCallback((file: File) => {
    const error = validate(file);
    if (error) {
      setValidationError(error);
      setSelectedFile(null);
      return;
    }
    setValidationError(null);
    setSelectedFile(file);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const handleUpload = () => {
    if (selectedFile) onFileSelect(selectedFile);
  };

  const handleClear = () => {
    setSelectedFile(null);
    setValidationError(null);
  };

  return (
    <div className="flex flex-col gap-4 w-full">
      {/* Drop target */}
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={cn(
          "relative flex flex-col items-center justify-center",
          "w-full h-48 rounded-xl border-2 border-dashed transition-all duration-200 cursor-pointer",
          "text-muted-foreground text-sm",
          isDragging
            ? "border-primary bg-primary/5 text-primary"
            : "border-border hover:border-primary/50 hover:bg-accent/30",
          (disabled || isPending) && "opacity-50 pointer-events-none"
        )}
      >
        <input
          type="file"
          accept=".pdf,.txt,.md,.doc,.docx"
          onChange={handleInputChange}
          className="absolute inset-0 opacity-0 cursor-pointer"
          disabled={disabled || isPending}
        />
        <Upload className="w-8 h-8 mb-3 opacity-60" />
        <p className="font-medium">Drop a file or click to browse</p>
        <p className="text-xs mt-1 opacity-60">PDF, TXT, MD, DOCX · Max {MAX_SIZE_MB}MB</p>
      </div>

      {/* Validation error */}
      {validationError && (
        <p className="text-sm text-destructive flex items-center gap-2">
          <X className="w-4 h-4" /> {validationError}
        </p>
      )}

      {/* Selected file preview */}
      {selectedFile && !validationError && (
        <div className="flex items-center justify-between px-4 py-3 rounded-lg bg-accent border border-border">
          <div className="flex items-center gap-3 min-w-0">
            <FileText className="w-5 h-5 text-primary shrink-0" />
            <div className="min-w-0">
              <p className="text-sm font-medium truncate">{selectedFile.name}</p>
              <p className="text-xs text-muted-foreground">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={handleClear}
            disabled={isPending}
            className="shrink-0 ml-2"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>
      )}

      {/* Upload button */}
      {selectedFile && !validationError && (
        <Button
          onClick={handleUpload}
          disabled={isPending || !selectedFile}
          className="w-full"
        >
          {isPending ? "Uploading..." : "Upload & Ingest"}
        </Button>
      )}
    </div>
  );
}


