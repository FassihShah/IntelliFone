"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { supabase } from "../lib/supabaseClient"
import Inbox from "../components/chat/Inbox"
import ChatWindow from "../components/chat/ChatWindow"

export default function ChatPage() {
  const router = useRouter()
  const [currentUser, setCurrentUser] = useState<any>(null)
  const [activeConversation, setActiveConversation] = useState<string | null>(null)

  useEffect(() => {
    // 1️⃣ Get current session
    const session = supabase.auth.getSession().then(({ data }) => {
      if (!data?.session?.user) {
        router.push("/signin")
      } else {
        setCurrentUser(data.session.user)
      }
    })

    // 2️⃣ Listen for auth changes
    const { data: listener } = supabase.auth.onAuthStateChange(
      (event, session) => {
        if (!session?.user) router.push("/signin")
        else setCurrentUser(session.user)
      }
    )

    return () => {
      listener.subscription.unsubscribe()
    }
  }, [])

  if (!currentUser) return null // avoid flicker / redirect before session loads

  return (
    <div className="chat-page flex h-full">
      <Inbox
        currentUserId={currentUser.id}
        activeConversation={activeConversation}
        setActiveConversation={setActiveConversation}
      />
      <ChatWindow
        currentUserId={currentUser.id}
        conversationId={activeConversation}
      />
    </div>
  )
}
