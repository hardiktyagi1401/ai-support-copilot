import { create } from "zustand";
import { devtools } from "zustand/middleware";
import type {
  RetrievalStats,
  LatencyBreakdown,
  HallucinationFlags,
} from "@/lib/hooks/use-ask";

// ---------------------------------------------------------------------------
// Re-export so components import from one place, not two
// ---------------------------------------------------------------------------

export type { RetrievalStats, LatencyBreakdown, HallucinationFlags };

// ---------------------------------------------------------------------------
// CitationSource — mirrors CitationSource in app/models/schemas.py exactly
// ---------------------------------------------------------------------------

export type CitationSource = {
  chunk_index: number;
  filename: string;
  page_number: number | null;
  relevance_score: number;
  text_preview: string;
};

// ---------------------------------------------------------------------------
// Message — the core unit of the chat thread
// All optional fields come from the assistant response only
// ---------------------------------------------------------------------------

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: CitationSource[];
  retrieval_stats?: RetrievalStats;
  latency?: LatencyBreakdown;
  hallucination_flags?: HallucinationFlags;
  citation_coverage?: number;
  isStreaming?: boolean;
  timestamp: Date;
};

// ---------------------------------------------------------------------------
// Store shape
// ---------------------------------------------------------------------------

type ChatState = {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  activeMessageId: string | null; // drives citation inspector panel in Phase 4

  addMessage: (message: Message) => void;
  updateLastAssistantMessage: (patch: Partial<Message>) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setActiveMessageId: (id: string | null) => void;
  clearChat: () => void;
};

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export const useChatStore = create<ChatState>()(
  devtools(
    (set) => ({
      messages: [],
      isLoading: false,
      error: null,
      activeMessageId: null,

      addMessage: (message) =>
        set(
          (state) => ({ messages: [...state.messages, message] }),
          false,
          "addMessage"
        ),

      updateLastAssistantMessage: (patch) =>
        set(
          (state) => {
            const messages = [...state.messages];
            const lastIdx = messages.findLastIndex(
              (m) => m.role === "assistant"
            );
            if (lastIdx === -1) return state;
            messages[lastIdx] = { ...messages[lastIdx], ...patch };
            return { messages };
          },
          false,
          "updateLastAssistantMessage"
        ),

      setLoading: (loading) =>
        set({ isLoading: loading }, false, "setLoading"),

      setError: (error) =>
        set({ error }, false, "setError"),

      setActiveMessageId: (id) =>
        set({ activeMessageId: id }, false, "setActiveMessageId"),

      clearChat: () =>
        set(
          { messages: [], error: null, activeMessageId: null },
          false,
          "clearChat"
        ),
    }),
    { name: "ChatStore" }
  )
);