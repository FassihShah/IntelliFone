"use client";
import { useEffect, useState } from "react";
import { supabase } from "../lib/supabaseClient";

interface ChatSidebarProps {
  onSelect: (conversationId: string) => void;
  userId: string;
  conversations: { id: string; mongo_conversation_id: string }[];
}

export default function ChatSidebar({ onSelect, userId }: ChatSidebarProps) {
  const [conversations, setConversations] = useState<any[]>([]);

  useEffect(() => {
    async function fetchConversations() {
      const { data, error } = await supabase
        .from("conversations")
        .select("*")
        .eq("user_id", userId)
        .order("created_at", { ascending: false });

      if (!error && data) setConversations(data);
    }
    if (userId) fetchConversations();
  }, [userId]);

  return (
    <div className="w-72 border-r border-zinc-800 bg-[#0f0f10] flex flex-col h-screen">
      <div className="p-6">
        <h1 className="text-2xl font-bold text-[#facc15] tracking-tight mb-6">IntelliFone</h1>
        <button
          className="w-full bg-[#facc15] hover:bg-[#eab308] text-black font-bold py-3 px-4 rounded-xl transition-all flex items-center justify-center gap-2 shadow-lg shadow-yellow-500/10"
          onClick={() => onSelect("")}
        >
          <span className="text-xl">+</span> New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-3 space-y-1">
        <p className="text-zinc-500 text-xs font-semibold uppercase tracking-widest px-3 mb-2">History</p>
        {conversations.map((conv) => (
          <div
            key={conv.id}
            className="group p-3 rounded-xl hover:bg-zinc-800/50 cursor-pointer transition-all border border-transparent hover:border-zinc-700"
            onClick={() => onSelect(conv.mongo_conversation_id)}
          >
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 rounded-full bg-zinc-600 group-hover:bg-[#facc15]" />
              <span className="text-sm text-zinc-300 group-hover:text-white truncate">
                Chat {conv.id.slice(0, 8)}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}