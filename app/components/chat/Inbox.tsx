"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { supabase } from "../../lib/supabaseClient";
import { useRouter } from "next/navigation";
import Pusher from "pusher-js";
import "./chat.css";

interface Message {
  sender_id: string;
  read_at: string | null;
  content?: string;
  created_at?: string;
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

function getInitials(name: string) {
  return name
    .split(" ")
    .map((w) => w[0])
    .join("")
    .substring(0, 2)
    .toUpperCase();
}

function formatTime(dateString?: string) {
  if (!dateString) return "";
  const d = new Date(dateString);
  const now = new Date();
  const isToday = d.toDateString() === now.toDateString();
  if (isToday) {
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  const yesterday = new Date(now);
  yesterday.setDate(yesterday.getDate() - 1);
  if (d.toDateString() === yesterday.toDateString()) return "Yesterday";
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

export default function Inbox({ currentUserId, activeConversation }: InboxProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [search, setSearch] = useState("");
  const router = useRouter();
  const pusherRef = useRef<Pusher | null>(null);

  const fetchInbox = useCallback(async () => {
    if (!currentUserId) return;

    const { data, error } = await supabase
      .from("conversation")
      .select(`id, user1_id, user2_id, messages ( sender_id, read_at, content, created_at )`)
      .or(`user1_id.eq.${currentUserId},user2_id.eq.${currentUserId}`);

    if (error) {
      console.error("Error fetching inbox:", error);
      return;
    }
    if (data) setConversations(data as Conversation[]);
  }, [currentUserId]);

  useEffect(() => {
    fetchInbox();

    if (!pusherRef.current) {
      pusherRef.current = new Pusher(process.env.NEXT_PUBLIC_PUSHER_KEY!, {
        cluster: process.env.NEXT_PUBLIC_PUSHER_CLUSTER!,
      });
    }

    const channel = pusherRef.current.subscribe(`inbox-${currentUserId}`);
    channel.bind("refresh-inbox", () => { fetchInbox(); });

    return () => {
      channel.unbind_all();
      pusherRef.current?.unsubscribe(`inbox-${currentUserId}`);
    };
  }, [currentUserId, fetchInbox]);

  const getOtherUserId = (convo: Conversation) =>
    convo.user1_id === currentUserId ? convo.user2_id : convo.user1_id;

  const getUnreadCount = (convo: Conversation) =>
    convo.messages.filter((m) => m.sender_id !== currentUserId && m.read_at === null).length;

  const getLastMessage = (convo: Conversation): Message | undefined => {
    const sorted = [...convo.messages].sort(
      (a, b) => new Date(b.created_at ?? 0).getTime() - new Date(a.created_at ?? 0).getTime()
    );
    return sorted[0];
  };

  const filtered = conversations.filter(() => true); // will be filtered by ConversationItem name

  return (
    <div className="inbox">
      <div className="inbox-header">
        <h2 className="inbox-title">Messages</h2>
        <div className="inbox-search-wrap">
          <svg className="inbox-search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            className="inbox-search"
            placeholder="Search conversations…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>

      <div className="inbox-list">
        {conversations.length === 0 && (
          <div className="no-messages">
            <span style={{ fontSize: "2rem" }}>💬</span>
            <span>No conversations yet</span>
          </div>
        )}
        {filtered.map((convo) => (
          <ConversationItem
            key={convo.id}
            otherUserId={getOtherUserId(convo)}
            unread={getUnreadCount(convo)}
            lastMessage={getLastMessage(convo)}
            isActive={activeConversation === convo.id}
            search={search}
            onClick={() => router.push(`/chats?conversation=${convo.id}`)}
          />
        ))}
      </div>
    </div>
  );
}

function ConversationItem({
  otherUserId,
  unread,
  lastMessage,
  isActive,
  search,
  onClick,
}: {
  otherUserId: string;
  unread: number;
  lastMessage?: Message;
  isActive: boolean;
  search: string;
  onClick: () => void;
}) {
  const [otherUserName, setOtherUserName] = useState("Loading…");

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

  // Filter by search
  if (search && !otherUserName.toLowerCase().includes(search.toLowerCase())) return null;

  const initials = otherUserName === "Loading…" ? "?" : getInitials(otherUserName);
  const preview = lastMessage?.content ?? "No messages yet";
  const time = formatTime(lastMessage?.created_at);

  return (
    <div className={`inbox-item ${isActive ? "active" : ""}`} onClick={onClick}>
      <div className="inbox-avatar">
        {initials}
        <span className="inbox-avatar-online" />
      </div>
      <div className="inbox-item-body">
        <div className="inbox-item-top">
          <span className="inbox-item-name">{otherUserName}</span>
          <span className="inbox-item-time">{time}</span>
        </div>
        <div className="inbox-item-preview">{preview}</div>
      </div>
      {unread > 0 && <span className="unread-badge">{unread}</span>}
    </div>
  );
}