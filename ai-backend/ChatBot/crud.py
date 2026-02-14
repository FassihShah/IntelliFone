from bson import ObjectId
from datetime import datetime
from ChatBot.db import conversations_col, messages_col
from models import ChatHistoryResponse, ChatMessage
from typing import List

def create_conversation(user_id, first_message):
    doc = {
        "user_id": user_id,
        "title": first_message[:40],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    result = conversations_col.insert_one(doc)
    return str(result.inserted_id)

def get_chat_history(conversation_id, limit=10):
    msgs = messages_col.find(
        {"conversation_id": ObjectId(conversation_id)}
    ).sort("created_at", 1).limit(limit)

    return [
        {"role": m["role"], "content": m["content"]}
        for m in msgs
    ]

def save_message(conversation_id, user_id, role, content):
    messages_col.insert_one({
        "conversation_id": ObjectId(conversation_id),
        "user_id": user_id,
        "role": role,
        "content": content,
        "created_at": datetime.utcnow()
    })

    conversations_col.update_one(
        {"_id": ObjectId(conversation_id)},
        {"$set": {"updated_at": datetime.utcnow()}}
    )
def get_chat_history_formatted(conversation_id: str) -> ChatHistoryResponse:
    """Fetch messages from MongoDB and return in frontend-friendly format."""
    # Fetch all messages for the conversation
    messages_cursor = messages_col.find({"conversation_id": ObjectId(conversation_id)}).sort("created_at", 1)
    
    messages: List[ChatMessage] = []
    
    for msg in messages_cursor:
        messages.append(ChatMessage(
            role=msg["role"],
            content=msg["content"]
        ))
    
    return ChatHistoryResponse(
        conversation_id=conversation_id,
        messages=messages
    )