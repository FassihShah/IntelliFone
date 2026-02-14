import { Suspense } from "react";
import ChatClient from "@/app/chats/ChatClient";

export default function ChatsPage() {
  return (
    <Suspense
      fallback={
        <div className="flex h-screen items-center justify-center">
          <p>Loading chats...</p>
        </div>
      }
    >
      <ChatClient />
    </Suspense>
  );
}
