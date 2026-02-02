"use client";

import { useEffect } from "react";
import Pusher from "pusher-js";

export function useRealtimeChat(
  conversationId: string | null,
  onNewMessage: (message: any) => void
) {
  useEffect(() => {
    if (!conversationId) return;

    const pusher = new Pusher(process.env.NEXT_PUBLIC_PUSHER_KEY!, {
      cluster: process.env.NEXT_PUBLIC_PUSHER_CLUSTER!,
    });

    const channel = pusher.subscribe(`chat-${conversationId}`);

    channel.bind("new-message", (data: any) => {
      onNewMessage(data);
    });

    return () => {
      channel.unbind_all();
      channel.unsubscribe();
      pusher.disconnect();
    };
  }, [conversationId, onNewMessage]);
}