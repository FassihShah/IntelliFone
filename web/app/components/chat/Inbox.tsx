"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { supabase } from "../../lib/supabaseClient";
import { useRouter } from "next/navigation";
import Pusher from "pusher-js";
import "./chat.css";

interface Message {
  sender_id: string;
  read_at: string | null;
}

interface Conversation {
  id: string;
  user1_id: string;
  user2_id: string;
  messages: Message[];
}

interface InboxProps {
  currentUserId: string;
  activeConversation?: string;
}

export default function Inbox({ currentUserId, activeConversation }: InboxProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const router = useRouter();
  
  // Use a ref to prevent multiple pusher instances during strict mode
  const pusherRef = useRef<Pusher | null>(null);

  const fetchInbox = useCallback(async () => {
    if (!currentUserId) return;
    
    const { data, error } = await supabase
      .from("conversation") // Ensure this matches your DB table name
      .select(`
        id,
        user1_id,
        user2_id,
        messages (
          sender_id,
          read_at
        )
      `)
      // Optional: Only fetch conversations the user is part of
      .or(`user1_id.eq.${currentUserId},user2_id.eq.${currentUserId}`);

    if (error) {
      console.error("Error fetching inbox:", error);
      return;
    }
    if (data) setConversations(data);
  }, [currentUserId]);

  useEffect(() => {
    fetchInbox();

    // Initialize Pusher only once
    if (!pusherRef.current) {
      pusherRef.current = new Pusher(process.env.NEXT_PUBLIC_PUSHER_KEY!, {
        cluster: process.env.NEXT_PUBLIC_PUSHER_CLUSTER!,
      });
    }

    // Subscribe to a SINGLE channel for this specific user
    const channel = pusherRef.current.subscribe(`inbox-${currentUserId}`);

    // When the API sends a "refresh-inbox" event, we just fetch data once
    channel.bind("refresh-inbox", () => {
      fetchInbox();
    });

    return () => {
      channel.unbind_all();
      pusherRef.current?.unsubscribe(`inbox-${currentUserId}`);
    };
  }, [currentUserId, fetchInbox]);

  // UI Helpers
  const getOtherUserId = (convo: Conversation) => 
    convo.user1_id === currentUserId ? convo.user2_id : convo.user1_id;

  const getUnreadCount = (convo: Conversation) => 
    convo.messages.filter(m => m.sender_id !== currentUserId && m.read_at === null).length;

  return (
    <div className="inbox">
      <h2 className="inbox-title">Inbox</h2>
      {conversations.length === 0 && (
        <div className="no-messages">No conversations yet</div>
      )}
      {conversations.map((convo) => (
        <ConversationItem
          key={convo.id}
          otherUserId={getOtherUserId(convo)}
          unread={getUnreadCount(convo)}
          isActive={activeConversation === convo.id}
          onClick={() => router.push(`/chats?conversation=${convo.id}`)}
        />
      ))}
    </div>
  );
}

function ConversationItem({ otherUserId, unread, isActive, onClick }: any) {
  const [otherUserName, setOtherUserName] = useState("Loading...");

  useEffect(() => {
    let mounted = true;
    const fetchName = async () => {
      const { data } = await supabase
        .from("profiles")
        .select("full_name")
        .eq("id", otherUserId)
        .single();
      if (mounted) setOtherUserName(data?.full_name || "Unknown User");
    };
    fetchName();
    return () => { mounted = false; };
  }, [otherUserId]);

  return (
    <div className={`inbox-item ${isActive ? "active" : ""}`} onClick={onClick}>
      <span className="username">{otherUserName}</span>
      {unread > 0 && <span className="unread-badge">{unread}</span>}
    </div>
  );
}