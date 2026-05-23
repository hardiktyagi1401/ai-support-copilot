import { AppShell } from "@/components/layout/app-shell";
import { UploadPanel } from "@/components/upload/upload-panel";

export default function DocumentsPage() {
  return (
    <AppShell>
      <div className="flex flex-col flex-1 overflow-y-auto p-8">
        <UploadPanel />
      </div>
    </AppShell>
  );
}