"use client";

import { useEffect, useRef } from "react";
import { BotMessageSquare, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ChatMessage } from "./chat-message";
import { ChatInput } from "./chat-input";
import { useChatStore } from "@/store/chat-store";
import { useAsk } from "@/lib/hooks/use-ask";

export function ChatPanel() {
  const { messages, isLoading, clearChat } = useChatStore();
  const { ask, isPending } = useAsk();
  const bottomRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom whenever a new message is added
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  const handleSubmit = (query: string) => {
    ask(query);
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* ---------------------------------------------------------------- */}
      {/* Header                                                            */}
      {/* ---------------------------------------------------------------- */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
        <div className="flex items-center gap-2">
          <BotMessageSquare className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm font-medium">Chat</span>
          {messages.length > 0 && (
            <span className="text-xs text-muted-foreground">
              · {messages.filter((m) => m.role === "user").length} question
              {messages.filter((m) => m.role === "user").length !== 1 ? "s" : ""}
            </span>
          )}
        </div>

        {messages.length > 0 && (
          <Button
            variant="ghost"
            size="icon"
            onClick={clearChat}
            disabled={isPending}
            className="h-7 w-7 text-muted-foreground hover:text-foreground"
            title="Clear chat"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </Button>
        )}
      </div>

      {/* ---------------------------------------------------------------- */}
      {/* Message list                                                      */}
      {/* ---------------------------------------------------------------- */}
      <div className="flex-1 overflow-y-auto min-h-0">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="py-2">
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            {/* Invisible anchor for scroll-to-bottom */}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* ---------------------------------------------------------------- */}
      {/* Input                                                             */}
      {/* ---------------------------------------------------------------- */}
      <ChatInput
        onSubmit={handleSubmit}
        isPending={isPending || isLoading}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Empty state — shown when no messages exist yet
// ---------------------------------------------------------------------------

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-4 px-8 text-center">
      <div className="flex items-center justify-center w-12 h-12 rounded-2xl bg-muted border border-border">
        <BotMessageSquare className="w-6 h-6 text-muted-foreground" />
      </div>
      <div className="flex flex-col gap-1">
        <p className="text-sm font-medium">Ask your documents anything</p>
        <p className="text-xs text-muted-foreground max-w-[280px] leading-relaxed">
          Upload a document first, then ask questions. Answers are grounded
          with citations from your knowledge base.
        </p>
      </div>
      <div className="flex flex-col gap-2 w-full max-w-xs">
        {EXAMPLE_QUERIES.map((q) => (
          <ExampleQuery key={q} query={q} />
        ))}
      </div>
    </div>
  );
}

const EXAMPLE_QUERIES = [
  "What are the main topics covered?",
  "Summarize the key findings",
  "What does the document say about…",
];

function ExampleQuery({ query }: { query: string }) {
  return (
    <div className="px-3 py-2 rounded-lg border border-border bg-muted/30 text-xs text-muted-foreground text-left cursor-default">
      {query}
    </div>
  );
}