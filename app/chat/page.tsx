"use client";

import { useEffect, useState } from "react";
import ChatSidebar from "../components/ChatSideBar";
import ChatWindow from "../components/ChatWindow";
import { supabase } from "../lib/supabaseClient";

export default function ChatPage() {
  const [userId, setUserId] = useState<string>("");
  const [selectedConversation, setSelectedConversation] = useState<string>("");
  const [conversations, setConversations] = useState<{ id: string; mongo_conversation_id: string }[]>([]);

  // 1️⃣ Get current user
  useEffect(() => {
    async function fetchUser() {
      const { data } = await supabase.auth.getUser();
      if (data.user) setUserId(data.user.id);
    }
    fetchUser();
  }, []);

  // 2️⃣ Fetch conversations for this user
  useEffect(() => {
    if (!userId) return;

    async function fetchConversations() {
      const { data, error } = await supabase
        .from("conversations")
        .select("*")
        .eq("user_id", userId)
        .order("created_at", { ascending: false });

      if (!error && data) setConversations(data as { id: string; mongo_conversation_id: string }[]);
    }

    fetchConversations();
  }, [userId]);

  // 3️⃣ Handle conversation selection or new chat
  const handleSelectConversation = async (mongoId: string) => {
    if (!mongoId) {
      // Create new conversation in backend
      const initialMessage = "Hi!"; // optional starter message
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          message: initialMessage,
          conversation_id: null, // null triggers backend to create
        }),
      });
      const data = await res.json();

      const newMongoId = data.conversation_id;

      // Save in Supabase
      const { error } = await supabase.from("conversations").insert([
        { user_id: userId, mongo_conversation_id: newMongoId },
      ]);

      if (!error) {
        setConversations([{ id: newMongoId, mongo_conversation_id: newMongoId }, ...conversations]);
        setSelectedConversation(newMongoId);
      }
    } else {
      // Existing conversation
      setSelectedConversation(mongoId);
    }
  };

  return (
    <div className="flex h-screen text-white bg-gray-900">
      <ChatSidebar
        onSelect={handleSelectConversation}
        userId={userId}
        conversations={conversations} // pass all conversations to sidebar
      />
      <ChatWindow
        userId={userId}
        conversationId={selectedConversation}
        onNewConversation={handleSelectConversation} // reuse same handler
      />
    </div>
  );
}
