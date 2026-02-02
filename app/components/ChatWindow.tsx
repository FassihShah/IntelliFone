"use client";
import { useEffect, useState, useRef } from "react";
import MessageInput from "./MessageInput";

export default function ChatWindow({ conversationId, userId, onNewConversation }: any) {
  const [messages, setMessages] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  useEffect(() => {
    async function fetchMessages() {
      if (!conversationId) return setMessages([]);
      setLoading(true);
      try {
        const res = await fetch(`/api/chat?conversation_id=${conversationId}`);
        const data = await res.json();
        if (Array.isArray(data.messages)) setMessages(data.messages);
      } catch (err) {
        console.error(err);
      }
      setLoading(false);
    }
    fetchMessages();
  }, [conversationId]);

  const sendMessage = async (msg: string) => {
    if (!msg.trim()) return;
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, message: msg, conversation_id: conversationId || null }),
      });
      const data = await res.json();
      if (!conversationId && onNewConversation && data.conversation_id) {
        onNewConversation(data.conversation_id);
      }
      setMessages((prev) => [...prev, { role: "user", content: msg }, { role: "assistant", content: data.reply }]);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-screen bg-[#09090b] text-zinc-100">
      {/* Header */}
      <div className="h-16 border-b border-zinc-800 flex items-center px-8 bg-[#09090b]/80 backdrop-blur-md sticky top-0 z-10">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse" />
          <span className="font-semibold text-zinc-200">AI Assistant</span>
        </div>
      </div>

      {/* Message Area */}
      <div className="flex-1 overflow-y-auto p-6 lg:px-24 space-y-6">
        {loading ? (
          <div className="flex justify-center py-10"><div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#facc15]"></div></div>
        ) : messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center space-y-4">
             <div className="w-16 h-16 bg-zinc-800 rounded-full flex items-center justify-center text-2xl">📱</div>
             <p className="text-zinc-500 max-w-xs">Ask me anything about phone specs, prices, or recommendations!</p>
          </div>
        ) : (
          messages.map((m, idx) => (
            <div key={idx} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`max-w-[80%] px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm ${
                m.role === "user" 
                  ? "bg-[#facc15] text-black font-medium rounded-tr-none" 
                  : "bg-zinc-800 text-zinc-100 rounded-tl-none border border-zinc-700"
              }`}>
                {m.content}
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input bar stays pinned */}
      <div className="p-6 lg:px-24 bg-gradient-to-t from-[#09090b] via-[#09090b] to-transparent">
        <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-2 focus-within:border-[#facc15]/50 transition-all shadow-2xl">
          <MessageInput onSend={sendMessage} />
        </div>
        <p className="text-[10px] text-zinc-600 text-center mt-3 uppercase tracking-tighter">Powered by IntelliFone Engine</p>
      </div>
    </div>
  );
}