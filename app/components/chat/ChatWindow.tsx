"use client"

import { useEffect, useState } from "react"
import { supabase } from "../../lib/supabaseClient"
import { useRealtimeChat } from "@/hooks/useRealtimeChat"
import "./chat.css"

interface Message {
  id: string
  sender_id: string
  content: string
  created_at: string
}

interface ChatWindowProps {
  currentUserId: string
  conversationId: string | null
}

export default function ChatWindow({
  currentUserId,
  conversationId,
}: ChatWindowProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [text, setText] = useState("")

  // Fetch messages whenever conversationId changes
  useEffect(() => {
    if (conversationId) {
      fetchMessages()
      markAsRead()
    } else {
      setMessages([])
    }
  }, [conversationId])

  // Fetch all messages for the conversation
  async function fetchMessages() {
    if (!conversationId) return

    const { data } = await supabase
      .from("messages")
      .select("*")
      .eq("conversation_id", conversationId)
      .order("created_at", { ascending: true })

    if (data) setMessages(data)
  }

  // Send a message
  async function sendMessage() {
    if (!text.trim() || !conversationId) return

    await supabase.from("messages").insert({
      conversation_id: conversationId,
      sender_id: currentUserId,
      content: text,
    })

    setText("")
  }

  // Mark all received messages as read
  async function markAsRead() {
    if (!conversationId) return

    await supabase
      .from("messages")
      .update({ read_at: new Date().toISOString() })
      .eq("conversation_id", conversationId)
      .is("read_at", null)
      .neq("sender_id", currentUserId)
  }

  // Realtime subscription for new messages
 if (conversationId) {
  useRealtimeChat(conversationId, msg =>
    setMessages(prev => [...prev, msg])
  )
}

  // No active conversation selected
  if (!conversationId) {
    return (
      <div className="chat-window empty">
        Select a conversation
      </div>
    )
  }

  return (
    <div className="chat-window">
      <div className="messages">
        {messages.length === 0 && (
          <div className="no-messages">Say hi 👋 and start the conversation</div>
        )}

        {messages.map(msg => (
          <div
            key={msg.id}
            className={`message ${
              msg.sender_id === currentUserId ? "outgoing" : "incoming"
            }`}
          >
            {msg.content}
          </div>
        ))}
      </div>

      <div className="input-bar">
        <input
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="Type a message..."
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  )
}
