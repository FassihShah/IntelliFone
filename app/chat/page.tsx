"use client";

import { useEffect, useState } from "react";
import ChatSidebar from "../components/ChatSideBar";
import ChatWindow from "../components/ChatWindow";
import { supabase } from "../lib/supabaseClient";

interface Conversation {
  id: string;
  mongo_conversation_id: string;
  title?: string;
}

export default function ChatPage() {
  const [userId, setUserId] = useState<string>("");
  const [selectedConversation, setSelectedConversation] = useState<string>("");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loadingConvs, setLoadingConvs] = useState(false);

  // 1️⃣ Get current user
  useEffect(() => {
    async function fetchUser() {
      const { data } = await supabase.auth.getUser();
      if (data.user) setUserId(data.user.id);
    }
    fetchUser();
  }, []);

  // 2️⃣ Fetch conversations from MongoDB via FastAPI
  const fetchConversations = async (uid: string) => {
    if (!uid) return;
    setLoadingConvs(true);
    try {
      const res = await fetch(`/api/chat?user_id=${encodeURIComponent(uid)}`);
      const data = await res.json();
      if (data.conversations) {
        setConversations(
          data.conversations.map((c: { conversation_id: string; title?: string }) => ({
            id: c.conversation_id,
            mongo_conversation_id: c.conversation_id,
            title: c.title,
          }))
        );
      }
    } catch (err) {
      console.error("Failed to fetch conversations:", err);
    } finally {
      setLoadingConvs(false);
    }
  };

  useEffect(() => {
    if (userId) fetchConversations(userId);
  }, [userId]);

  // 3️⃣ Handle conversation selection or new chat
  const handleSelectConversation = async (mongoId: string) => {
    if (!mongoId) {
      // New chat — clear selection
      setSelectedConversation("");
      return;
    }
    setSelectedConversation(mongoId);

    // Optimistically add to sidebar if not already present
    setConversations((prev) => {
      if (prev.some((c) => c.mongo_conversation_id === mongoId)) return prev;
      return [{ id: mongoId, mongo_conversation_id: mongoId }, ...prev];
    });
  };

  // 4️⃣ Called by ChatWindow after a new conversation is created via streaming
  const handleNewConversation = async (mongoId: string) => {
    setSelectedConversation(mongoId);
    // Re-fetch so we get the real title from MongoDB
    await fetchConversations(userId);
  };

  return (
    <div className="flex h-screen text-white bg-gray-900">
      <ChatSidebar
        onSelect={handleSelectConversation}
        userId={userId}
        conversations={conversations}
        loading={loadingConvs}
        selectedConversationId={selectedConversation}
      />
      <ChatWindow
        userId={userId}
        conversationId={selectedConversation}
        onNewConversation={handleNewConversation}
      />
    </div>
  );
}