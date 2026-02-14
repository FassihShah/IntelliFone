import Pusher from "pusher";
import { NextResponse } from "next/server";

const pusher = new Pusher({
  appId: process.env.PUSHER_APP_ID!,
  key: process.env.NEXT_PUBLIC_PUSHER_KEY!,
  secret: process.env.PUSHER_SECRET!,
  cluster: process.env.NEXT_PUBLIC_PUSHER_CLUSTER!,
  useTLS: true,
});

export async function POST(req: Request) {
  try {
    const { conversationId, senderId, recipientId, content } = await req.json();

    const messageData = {
      id: Math.random().toString(36).substr(2, 9), 
      sender_id: senderId,
      content,
      created_at: new Date().toISOString(),
    };

    // 1️⃣ Trigger the specific chat room (updates ChatWindow)
    await pusher.trigger(`chat-${conversationId}`, "new-message", messageData);

    // 2️⃣ Trigger the recipient's inbox (updates Inbox list/badges)
    if (recipientId) {
      await pusher.trigger(`inbox-${recipientId}`, "refresh-inbox", {});
    }
    
    // 3️⃣ Optional: Trigger sender's inbox too (to update their own "last message" view)
    await pusher.trigger(`inbox-${senderId}`, "refresh-inbox", {});

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Pusher Error:", error);
    return NextResponse.json({ error: "Pusher trigger failed" }, { status: 500 });
  }
}