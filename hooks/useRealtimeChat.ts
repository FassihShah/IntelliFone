"use client";

import { useEffect } from "react"
import { supabase } from "../app/lib/supabaseClient"

export function useRealtimeChat(
  conversationId: string | null,
  onNewMessage: (message: any) => void
) {
  useEffect(() => {
    if (!conversationId) return // guard null safely

    const channel = supabase
      .channel(`chat-${conversationId}`)
      .on(
        "postgres_changes",
        {
          event: "INSERT",
          schema: "public",
          table: "messages",
          filter: `conversation_id=eq.${conversationId}`
        },
        payload => onNewMessage(payload.new)
      )
      .subscribe()

    // Cleanup
    return () => {
      supabase.removeChannel(channel)
    }
  }, [conversationId, onNewMessage])
}
