"use client"

import { useEffect, useState } from "react"
import { supabase } from "../../lib/supabaseClient"
import { useRouter } from "next/navigation"
import "./chat.css"

interface Message {
  sender_id: string
  read_at: string | null
}

interface Conversation {
  id: string
  user1_id: string
  user2_id: string
  messages: Message[]
}

interface InboxProps {
  currentUserId: string
  activeConversation: string | null
  setActiveConversation: (id: string) => void
}

export default function Inbox({
  currentUserId,
  activeConversation,
  setActiveConversation,
}: InboxProps) {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [usernames, setUsernames] = useState<Record<string, string>>({})
  const router = useRouter()

  // Fetch inbox on mount
  useEffect(() => {
    fetchInbox()
  }, [])

  // Subscribe to Realtime changes for new conversations
  useEffect(() => {
    const channel = supabase
      .channel("conversation-inbox")
      .on(
        "postgres_changes",
        {
          event: "INSERT",
          schema: "public",
          table: "conversation",
        },
        payload => {
          const convo = payload.new
          if (convo.user1_id === currentUserId || convo.user2_id === currentUserId) {
            fetchInbox()
          }
        }
      )
      .subscribe()

    // Synchronous cleanup (no async)
    return () => {
      supabase.removeChannel(channel)
    }
  }, [currentUserId])

  // Fetch inbox from Supabase
  async function fetchInbox() {
    const { data } = await supabase
      .from("conversation")
      .select(`
        id,
        user1_id,
        user2_id,
        messages (
          sender_id,
          read_at
        )
      `)
      .order("id", { ascending: false })

    if (data) {
      setConversations(data)
      fetchUsernames(data)
    }
  }

  // Pre-fetch usernames for all conversations
  async function fetchUsernames(convos: Conversation[]) {
    const ids = convos.map(c => (c.user1_id === currentUserId ? c.user2_id : c.user1_id))
    const uniqueIds = Array.from(new Set(ids))

    const { data } = await supabase
      .from("profiles")
      .select("id, full_name")
      .in("id", uniqueIds)

    if (data) {
      const map: Record<string, string> = {}
      data.forEach(u => (map[u.id] = u.full_name))
      setUsernames(map)
    }
  }

  // Get the other user in a conversation
  function getOtherUser(convo: Conversation) {
    return convo.user1_id === currentUserId ? convo.user2_id : convo.user1_id
  }

  // Count unread messages
  function getUnreadCount(convo: Conversation) {
    return convo.messages.filter(
      m => m.sender_id !== currentUserId && m.read_at === null
    ).length
  }

  return (
    <div className="inbox">
      <h2 className="inbox-title">Inbox</h2>

      {conversations.map(convo => {
        const otherUserId = getOtherUser(convo)
        const unread = getUnreadCount(convo)
        const name = usernames[otherUserId] || "Loading..."

        return (
          <div
            key={convo.id}
            className={`inbox-item ${
              activeConversation === convo.id ? "active" : ""
            }`}
            onClick={() => {
              setActiveConversation(convo.id)
              router.push(`/chats?conversation=${convo.id}`)
            }}
          >
            <span className="username">{name}</span>
            {unread > 0 && <span className="unread-badge">{unread}</span>}
          </div>
        )
      })}
    </div>
  )
}
