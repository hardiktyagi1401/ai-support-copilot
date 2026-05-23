"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Bot, User, ChevronDown, ChevronUp, FileText, AlertTriangle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Message, CitationSource } from "@/store/chat-store";

// ---------------------------------------------------------------------------
// Citation chip — inline [Src N] marker rendered as an interactive badge
// ---------------------------------------------------------------------------

function CitationChip({
  n,
  citation,
  isActive,
  onClick,
}: {
  n: number;
  citation: CitationSource | undefined;
  isActive: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      title={citation?.filename ?? `Source ${n}`}
      className={cn(
        "inline-flex items-center justify-center",
        "mx-0.5 px-1.5 py-0 rounded text-[10px] font-semibold",
        "border transition-colors duration-150 cursor-pointer",
        "align-middle leading-5",
        isActive
          ? "bg-primary text-primary-foreground border-primary"
          : "bg-muted text-muted-foreground border-border hover:border-primary hover:text-primary"
      )}
    >
      {n}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Parse answer text — replace [Src N] with CitationChip components
// ---------------------------------------------------------------------------

function parseAnswerWithCitations(
  text: string,
  citations: CitationSource[],
  activeCitation: number | null,
  onCitationClick: (n: number) => void
): React.ReactNode[] {
  // Split on [Src N] markers, keeping the delimiters
  const parts = text.split(/(\[Src\s*\d+\])/gi);

  return parts.map((part, i) => {
    const match = part.match(/\[Src\s*(\d+)\]/i);
    if (match) {
      const n = parseInt(match[1], 10);
      const citation = citations[n - 1]; // 1-indexed in text, 0-indexed in array
      return (
        <CitationChip
          key={i}
          n={n}
          citation={citation}
          isActive={activeCitation === n}
          onClick={() => onCitationClick(n)}
        />
      );
    }
    return <span key={i}>{part}</span>;
  });
}

// ---------------------------------------------------------------------------
// Citation drawer — expands below the message when a chip is clicked
// ---------------------------------------------------------------------------

function CitationDrawer({
  citations,
  activeCitationN,
}: {
  citations: CitationSource[];
  activeCitationN: number | null;
}) {
  if (activeCitationN === null) return null;

  const citation = citations[activeCitationN - 1];
  if (!citation) return null;

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: "auto" }}
      exit={{ opacity: 0, height: 0 }}
      transition={{ duration: 0.18, ease: "easeOut" }}
      className="mt-3 rounded-lg border border-border bg-muted/50 overflow-hidden"
    >
      <div className="px-4 py-3 flex flex-col gap-1.5">
        <div className="flex items-center gap-2">
          <FileText className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
          <span className="text-xs font-semibold truncate">
            {citation.filename}
          </span>
          {citation.page_number && (
            <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-4">
              p. {citation.page_number}
            </Badge>
          )}
          <Badge variant="secondary" className="text-[10px] px-1.5 py-0 h-4 ml-auto shrink-0">
            {Math.round(citation.relevance_score * 100)}% match
          </Badge>
        </div>
        <p className="text-xs text-muted-foreground leading-relaxed line-clamp-4">
          {citation.text_preview}
        </p>
      </div>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Streaming skeleton — three animated dots while waiting
// ---------------------------------------------------------------------------

function StreamingSkeleton() {
  return (
    <div className="flex items-center gap-1 py-1">
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          className="w-2 h-2 rounded-full bg-muted-foreground/40"
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{
            duration: 1.2,
            repeat: Infinity,
            delay: i * 0.2,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ChatMessage — main export
// ---------------------------------------------------------------------------

export function ChatMessage({ message }: { message: Message }) {
  const [activeCitationN, setActiveCitationN] = useState<number | null>(null);
  const [showStats, setShowStats] = useState(false);

  const isUser = message.role === "user";
  const citations = message.citations ?? [];

  const handleCitationClick = (n: number) => {
    setActiveCitationN((prev) => (prev === n ? null : n));
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, ease: "easeOut" }}
      className={cn(
        "flex gap-3 px-4 py-3 group",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "flex items-start justify-center w-8 h-8 rounded-full shrink-0 mt-0.5",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted text-muted-foreground border border-border"
        )}
      >
        {isUser ? (
          <User className="w-4 h-4 mt-2" />
        ) : (
          <Bot className="w-4 h-4 mt-2" />
        )}
      </div>

      {/* Bubble */}
      <div
        className={cn(
          "flex flex-col gap-2 max-w-[75%]",
          isUser ? "items-end" : "items-start"
        )}
      >
        <div
          className={cn(
            "rounded-2xl px-4 py-3 text-sm leading-relaxed",
            isUser
              ? "bg-primary text-primary-foreground rounded-tr-sm"
              : "bg-muted text-foreground rounded-tl-sm border border-border"
          )}
        >
          {message.isStreaming ? (
            <StreamingSkeleton />
          ) : isUser ? (
            message.content
          ) : (
            parseAnswerWithCitations(
              message.content,
              citations,
              activeCitationN,
              handleCitationClick
            )
          )}
        </div>

        {/* Citation drawer */}
        {!isUser && citations.length > 0 && (
          <div className="w-full max-w-full">
            <CitationDrawer
              citations={citations}
              activeCitationN={activeCitationN}
            />
          </div>
        )}

        {/* Footer — citation count + latency stats toggle */}
        {!isUser && !message.isStreaming && (
          <div className="flex items-center gap-3 px-1">
            {citations.length > 0 && (
              <span className="text-[11px] text-muted-foreground">
                {citations.length} source{citations.length !== 1 ? "s" : ""}
              </span>
            )}

            {message.latency && (
              <button
                onClick={() => setShowStats((p) => !p)}
                className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors"
              >
                {message.latency.total_ms}ms
                {showStats ? (
                  <ChevronUp className="w-3 h-3" />
                ) : (
                  <ChevronDown className="w-3 h-3" />
                )}
              </button>
            )}

            {message.hallucination_flags?.any_flag_raised && (
              <span
                title="Hallucination signals detected — review citations"
                className="flex items-center gap-1 text-[11px] text-amber-500"
              >
                <AlertTriangle className="w-3 h-3" />
                Review
              </span>
            )}
          </div>
        )}

        {/* Latency breakdown — shown on demand */}
        {showStats && message.latency && message.retrieval_stats && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            transition={{ duration: 0.15 }}
            className="w-full rounded-lg border border-border bg-card px-3 py-2.5"
          >
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
              <StatRow label="Retrieval" value={`${message.latency.retrieval_ms}ms`} />
              <StatRow label="LLM" value={`${message.latency.llm_ms}ms`} />
              <StatRow label="Total" value={`${message.latency.total_ms}ms`} />
              <StatRow
                label="Chunks used"
                value={`${message.retrieval_stats.chunks_passed_threshold} / ${message.retrieval_stats.chunks_retrieved}`}
              />
              {message.retrieval_stats.top_similarity_score !== null && (
                <StatRow
                  label="Top score"
                  value={`${Math.round((message.retrieval_stats.top_similarity_score ?? 0) * 100)}%`}
                />
              )}
              {message.citation_coverage !== undefined && (
                <StatRow
                  label="Citation coverage"
                  value={`${Math.round(message.citation_coverage * 100)}%`}
                />
              )}
            </div>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <>
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium tabular-nums">{value}</span>
    </>
  );
}