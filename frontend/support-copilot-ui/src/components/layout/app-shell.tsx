"use client";

import { BotMessageSquare, FileStack, Settings } from "lucide-react";
import { cn } from "@/lib/utils";

type NavItem = {
  label: string;
  icon: React.ReactNode;
  href: string;
  disabled?: boolean;
};

const NAV_ITEMS: NavItem[] = [
  {
    label: "Chat",
    icon: <BotMessageSquare className="w-5 h-5" />,
    href: "/",
  },
  {
    label: "Documents",
    icon: <FileStack className="w-5 h-5" />,
    href: "/documents",
  },
  {
    label: "Settings",
    icon: <Settings className="w-5 h-5" />,
    href: "/settings",
    disabled: true,
  },
];

export function AppShell({ children }: { children?: React.ReactNode }) {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      {/* ------------------------------------------------------------------ */}
      {/* Sidebar                                                              */}
      {/* ------------------------------------------------------------------ */}
      <aside className="flex flex-col w-16 md:w-56 border-r border-border bg-card shrink-0 transition-all duration-200">
        {/* Logo / Brand */}
        <div className="flex items-center gap-3 px-4 py-5 border-b border-border">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary">
            <BotMessageSquare className="w-4 h-4 text-primary-foreground" />
          </div>
          <span className="hidden md:block text-sm font-semibold tracking-tight truncate">
            Support Copilot
          </span>
        </div>

        {/* Nav */}
        <nav className="flex flex-col gap-1 p-2 flex-1">
    {NAV_ITEMS.map((item) => (
    <a
      key={item.href}
      href={item.href}
      className={cn(
        "flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors",
        "text-muted-foreground hover:text-foreground hover:bg-accent",
        item.disabled && "opacity-40 pointer-events-none"
      )}
    >
      {item.icon}
      <span className="hidden md:block">{item.label}</span>
    </a>
   ))}
   </nav>
        {/* Footer */}
        <div className="p-4 border-t border-border hidden md:block">
          <p className="text-xs text-muted-foreground">RAG-powered · v0.1</p>
        </div>
      </aside>

      {/* ------------------------------------------------------------------ */}
      {/* Main content area                                                    */}
      {/* ------------------------------------------------------------------ */}
      <main className="flex flex-col flex-1 min-w-0 overflow-hidden">
        {children ?? (
          <div className="flex flex-1 items-center justify-center text-muted-foreground text-sm">
            Select a section to get started
          </div>
        )}
      </main>
    </div>
  );
}