import { AppShell } from "@/components/layout/app-shell";
import { ChatPanel } from "@/components/chat/chat-panel";

export default function HomePage() {
  return (
    <AppShell>
      <ChatPanel />
    </AppShell>
  );
}