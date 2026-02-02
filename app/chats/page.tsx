"use client";

import { useState, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { supabase } from "../lib/supabaseClient";
import Inbox from "../components/chat/Inbox";
import ChatWindow from "../components/chat/ChatWindow";

export default function ChatPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  
  // Get the conversation ID from the URL (?conversation=ID)
  const activeConversation = searchParams.get("conversation");

  const [currentUser, setCurrentUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkSession = async () => {
      const { data } = await supabase.auth.getSession();
      
      if (!data?.session) {
        router.push("/signin");
      } else {
        setCurrentUser(data.session.user);
      }
      setLoading(false);
    };

    checkSession();

    const { data: listener } = supabase.auth.onAuthStateChange(
      (event, session) => {
        if (event === "SIGNED_OUT" || !session) {
          router.push("/signin");
        } else {
          setCurrentUser(session.user);
          setLoading(false);
        }
      }
    );

    return () => {
      listener.subscription.unsubscribe();
    };
  }, [router]);

  // 🛡️ Guard: Show a loader instead of 'null' 
  // This prevents the "flash" that often triggers auto-logouts
  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <p>Loading your messages...</p>
      </div>
    );
  }

  if (!currentUser) return null;

  return (
    <div className="chat-page flex h-screen overflow-hidden">
      <div className="w-1/3 border-r h-full overflow-y-auto">
        <Inbox 
          currentUserId={currentUser.id} 
          activeConversation={activeConversation || undefined} 
        />
      </div>
      <div className="w-2/3 h-full">
        <ChatWindow
          currentUserId={currentUser.id}
          conversationId={activeConversation}
        />
      </div>
    </div>
  );
}