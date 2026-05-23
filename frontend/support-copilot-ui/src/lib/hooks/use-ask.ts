import { useMutation } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { useChatStore, type Message, type CitationSource } from "@/store/chat-store";
import { toast } from "sonner";

// ---------------------------------------------------------------------------
// Request — mirrors AskRequest in app/models/schemas.py
// ---------------------------------------------------------------------------

export type AskPayload = {
  query: string;
  tenant_id: string;
  top_k?: number;
  score_threshold?: number;
  document_ids?: string[];
  conversation_history?: { role: "user" | "assistant"; content: string }[];
};

// ---------------------------------------------------------------------------
// Response sub-types — mirrors AskResponse and nested schemas exactly
// ---------------------------------------------------------------------------

export type RetrievalStats = {
  chunks_retrieved: number;
  chunks_passed_threshold: number;
  top_similarity_score: number | null;
  score_threshold_used: number;
  embed_ms: number;
  search_ms: number;
};

export type LatencyBreakdown = {
  retrieval_ms: number;
  prompt_assembly_ms: number;
  llm_ms: number;
  total_ms: number;
};

export type HallucinationFlags = {
  answer_contains_no_citation: boolean;
  low_term_overlap: boolean;
  suspiciously_long_answer: boolean;
  no_chunks_passed_threshold: boolean;
  any_flag_raised: boolean;
};

export type AskResponse = {
  query_id: string;
  answered: boolean;
  answer: string | null;
  refusal_reason: string | null;
  citations: CitationSource[];
  hallucination_flags: HallucinationFlags;
  retrieval_stats: RetrievalStats;
  latency: LatencyBreakdown;
  citation_coverage: number;
  uncited_sentence_count: number;
  model_used: string;
  prompt_tokens: number | null;
  completion_tokens: number | null;
};

// ---------------------------------------------------------------------------
// Tenant — replace with real auth context when you add auth
// ---------------------------------------------------------------------------

const DEFAULT_TENANT_ID = "demo-tenant";

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useAsk() {
  const { addMessage, setLoading, setError, updateLastAssistantMessage } =
    useChatStore();

  const mutation = useMutation({
    mutationFn: async (payload: AskPayload): Promise<AskResponse> => {
      const { data } = await apiClient.post<AskResponse>("/api/v1/query/ask", payload);
      return data;
    },

    onMutate: (payload) => {
      // Immediately render user message — optimistic, no round-trip needed
      const userMessage: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: payload.query,
        timestamp: new Date(),
      };
      addMessage(userMessage);

      // Skeleton assistant message — replaced on success/error
      const assistantPlaceholder: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "",
        isStreaming: true,
        timestamp: new Date(),
      };
      addMessage(assistantPlaceholder);

      setLoading(true);
      setError(null);
    },

    onSuccess: (data) => {
      if (!data.answered) {
        // Backend withheld answer — low retrieval confidence or no chunks passed
        updateLastAssistantMessage({
          content:
            data.refusal_reason ??
            "I couldn't find relevant information to answer that.",
          citations: [],
          isStreaming: false,
          retrieval_stats: data.retrieval_stats,
          latency: data.latency,
          hallucination_flags: data.hallucination_flags,
        });
      } else {
        updateLastAssistantMessage({
          content: data.answer ?? "",
          citations: data.citations,
          isStreaming: false,
          retrieval_stats: data.retrieval_stats,
          latency: data.latency,
          hallucination_flags: data.hallucination_flags,
          citation_coverage: data.citation_coverage,
        });
      }
      setLoading(false);
    },

    onError: (error: Error) => {
      updateLastAssistantMessage({
        content: "Something went wrong. Please try again.",
        isStreaming: false,
      });
      setLoading(false);
      setError(error.message);
      toast.error("Request failed", { description: error.message });
    },
  });

  return {
    // Wraps mutate so callers don't need to pass tenant_id manually
    ask: (query: string, options?: Partial<AskPayload>) =>
      mutation.mutate({
        query,
        tenant_id: DEFAULT_TENANT_ID,
        ...options,
      }),
    isPending: mutation.isPending,
    isError: mutation.isError,
    reset: mutation.reset,
  };
}