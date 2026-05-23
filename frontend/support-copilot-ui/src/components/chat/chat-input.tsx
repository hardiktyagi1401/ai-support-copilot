"use client";

import { useState, useRef, useCallback } from "react";
import { Send, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type ChatInputProps = {
  onSubmit: (query: string) => void;
  isPending: boolean;
  disabled?: boolean;
};

export function ChatInput({ onSubmit, isPending, disabled }: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const canSubmit = value.trim().length > 0 && !isPending && !disabled;

  const handleSubmit = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || isPending) return;
    onSubmit(trimmed);
    setValue("");
    // Reset textarea height after submit
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [value, isPending, onSubmit]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Auto-grow textarea up to ~6 lines
  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 144)}px`; // 144px ≈ 6 lines
  };

  return (
    <div className="border-t border-border bg-background px-4 py-3">
      <div
        className={cn(
          "flex items-end gap-2 rounded-xl border border-border bg-muted/40",
          "px-3 py-2 transition-colors",
          "focus-within:border-primary/60 focus-within:bg-background",
          (disabled || isPending) && "opacity-60"
        )}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your documents…"
          disabled={disabled || isPending}
          rows={1}
          className={cn(
            "flex-1 resize-none bg-transparent text-sm outline-none",
            "placeholder:text-muted-foreground leading-relaxed",
            "min-h-[24px] max-h-[144px] py-0.5",
            "disabled:cursor-not-allowed"
          )}
        />

        <Button
          size="icon"
          onClick={handleSubmit}
          disabled={!canSubmit}
          className={cn(
            "h-8 w-8 shrink-0 rounded-lg transition-all",
            canSubmit
              ? "opacity-100"
              : "opacity-40 cursor-not-allowed"
          )}
        >
          {isPending ? (
            <Square className="w-3.5 h-3.5 fill-current" />
          ) : (
            <Send className="w-3.5 h-3.5" />
          )}
        </Button>
      </div>

      <p className="text-[11px] text-muted-foreground mt-1.5 px-1">
        Enter to send · Shift+Enter for new line
      </p>
    </div>
  );
}