"use client";

import { useEffect, useState } from "react";
import { supabase } from "../lib/supabaseClient";

interface ChatSidebarProps {
  onSelect: (conversationId: string) => void;
  userId: string;
  conversations: { id: string; mongo_conversation_id: string }[]; // Add this line
}

interface Conversation {
  id: string;
  mongo_conversation_id: string;
  created_at: string;
}

export default function ChatSidebar({ onSelect, userId }: ChatSidebarProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);

  useEffect(() => {
    async function fetchConversations() {
      const { data, error } = await supabase
        .from("conversations")
        .select("*")
        .eq("user_id", userId)
        .order("created_at", { ascending: false });

      if (!error && data) setConversations(data as Conversation[]);
    }
    if (userId) fetchConversations();
  }, [userId]);

  return (
    <div className="w-64 border-r border-gray-700 p-2 flex flex-col gap-2 h-screen overflow-y-auto">
      <button
        className="bg-blue-500 text-white py-2 rounded"
        onClick={() => onSelect("")} // empty string for new chat
      >
        + New Chat
      </button>

      {conversations.map((conv) => (
        <div
          key={conv.id}
          className="p-2 rounded hover:bg-gray-700 cursor-pointer"
          onClick={() => onSelect(conv.mongo_conversation_id)}
        >
          Chat {conv.id.slice(0, 6)}
        </div>
      ))}
    </div>
  );
}
