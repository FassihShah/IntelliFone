"use client";

import { useEffect, useState, useRef } from "react";
import MessageInput from "./MessageInput";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface ChatWindowProps {
  conversationId: string; // MongoDB conversation ID
  userId: string;
  onNewConversation?: (mongoId: string) => void;
}

export default function ChatWindow({ conversationId, userId, onNewConversation }: ChatWindowProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom whenever messages change
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  // Fetch messages when conversation changes
  useEffect(() => {
    async function fetchMessages() {
      if (!conversationId) return setMessages([]);
      setLoading(true);

      try {
        const res = await fetch(`/api/chat?conversation_id=${conversationId}`);
        const data = await res.json();

        // Ensure data.messages exists and is an array
        if (Array.isArray(data.messages)) {
          setMessages(data.messages);
        } else {
          setMessages([]);
        }
      } catch (err) {
        console.error("Failed to fetch messages:", err);
        setMessages([]);
      }

      setLoading(false);
    }

    fetchMessages();
  }, [conversationId]);

  // Send message
  const sendMessage = async (msg: string) => {
    if (!msg.trim()) return;

    const payload = {
      user_id: userId,
      message: msg,
      conversation_id: conversationId || null,
    };

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      // If it's a new conversation, notify parent
      if (!conversationId && onNewConversation && data.conversation_id) {
        onNewConversation(data.conversation_id);
      }

      // Update messages locally
      setMessages((prev) => [
        ...prev,
        { role: "user", content: msg },
        { role: "assistant", content: data.reply },
      ]);
    } catch (err) {
      console.error("Failed to send message:", err);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-screen bg-gray-900 text-white">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {loading ? (
          <p className="text-gray-400">Loading...</p>
        ) : messages.length === 0 ? (
          <p className="text-gray-400">Start a conversation!</p>
        ) : (
          messages.map((m, idx) => (
            <div
              key={idx}
              className={`p-2 rounded max-w-xs break-words ${
                m.role === "user"
                  ? "bg-blue-500 text-white self-end"
                  : "bg-gray-700 text-white self-start"
              }`}
            >
              {m.content}
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-700 p-2">
        <MessageInput onSend={sendMessage} />
      </div>
    </div>
  );
}
