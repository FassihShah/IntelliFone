"use client";

import { useEffect, useState, useCallback } from "react";
import { supabase } from "../../lib/supabaseClient";
import { useRealtimeChat } from "@/hooks/useRealtimeChat";
import "./chat.css";

interface Message {
  id: string;
  sender_id: string;
  content: string;
  created_at: string;
}

interface ChatWindowProps {
  currentUserId: string;
  conversationId: string | null;
}

export default function ChatWindow({ currentUserId, conversationId }: ChatWindowProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [text, setText] = useState("");
  const [recipientId, setRecipientId] = useState<string | null>(null); // Track the other person

  const handleIncomingMessage = useCallback((msg: Message) => {
    setMessages((prev) => {
      const exists = prev.find((m) => m.id === msg.id || (m.content === msg.content && m.created_at === msg.created_at));
      return exists ? prev : [...prev, msg];
    });
  }, []);

  useRealtimeChat(conversationId, handleIncomingMessage);

  // Fetch both messages AND the conversation details to find the recipient
  async function fetchData() {
    if (!conversationId) return;

    // 1. Get messages
    const { data: msgData } = await supabase
      .from("messages")
      .select("*")
      .eq("conversation_id", conversationId)
      .order("created_at", { ascending: true });
    
    if (msgData) setMessages(msgData);

    // 2. Get conversation to find the recipientId
    const { data: convoData } = await supabase
      .from("conversation")
      .select("user1_id, user2_id")
      .eq("id", conversationId)
      .single();

    if (convoData) {
      // If I am user1, the recipient is user2, and vice versa.
      const otherId = convoData.user1_id === currentUserId ? convoData.user2_id : convoData.user1_id;
      setRecipientId(otherId);
    }
  }

  useEffect(() => {
    fetchData();
  }, [conversationId, currentUserId]);

  async function sendMessage() {
    if (!text.trim() || !conversationId || !recipientId) return;

    const msgContent = text;
    setText(""); 

    // 1. Save to Database
    await supabase.from("messages").insert({
      conversation_id: conversationId,
      sender_id: currentUserId,
      content: msgContent,
    });

    // 2. Push to Pusher API (now including recipientId)
    await fetch("/api/messages/send", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversationId,
        senderId: currentUserId,
        recipientId: recipientId, // Important: This triggers the recipient's Inbox update
        content: msgContent,
      }),
    });
  }

  if (!conversationId) {
    return <div className="chat-window empty">Select a conversation</div>;
  }

  return (
    <div className="chat-window">
      <div className="messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.sender_id === currentUserId ? "outgoing" : "incoming"}`}>
            {msg.content}
          </div>
        ))}
      </div>
      <div className="input-bar">
        <input 
          value={text} 
          onChange={(e) => setText(e.target.value)} 
          placeholder="Type a message..." 
          onKeyDown={(e) => e.key === 'Enter' && sendMessage()} 
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}