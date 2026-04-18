import os
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from RecommendationEngine.recommendation_service import get_recommendations, stream_recommendations


SYSTEM_PROMPT = """
You are a mobile phone expert assistant.

You ONLY answer questions related to:
- smartphones
- mobile hardware or software
- buying/selling used phones
- mobile diagnostics and pricing

If the question is unrelated, politely refuse.
use -> for headings.

Use **bold text** for phone names, prices, and key points.

Use *italic text* only for emphasis, not headings.

Present lists  numbered lists where appropriate.

Ensure consistent spacing and clean line breaks between sections.

Do not use * .

Avoid emojis; keep the tone professional and informative.

Keep explanations concise and structured in short paragraphs.
"""


def create_llm():
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=0.3,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


def extract_budget_and_priority(message: str):
    """
    Extract budget and priority from a recommendation-style user message.
    """
    budget_match = re.search(r"(\d{2,6})", message.replace(",", ""))
    max_price = float(budget_match.group(1)) if budget_match else 70000

    priorities = {
        "gaming": ["gaming", "performance", "fps"],
        "camera": ["camera", "photography", "selfie"],
        "battery": ["battery", "backup", "mah"],
        "general": ["all round", "balanced", "daily use"],
    }

    priority = "general"
    message_lower = message.lower()

    for key, words in priorities.items():
        if any(word in message_lower for word in words):
            priority = key
            break

    return max_price, priority


def is_recommendation_query(message: str) -> bool:
    keywords = [
        "recommend",
        "recommendation",
        "suggest",
        "suggestion",
        "best phone",
        "which phone",
        "buy",
        "purchase",
    ]

    message = message.lower()
    return any(keyword in message for keyword in keywords)


def build_messages(chat_history, user_message):
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_message))
    return messages


async def generate_stream_reply(chat_history, user_message):
    """
    Stream chatbot output. Recommendation queries stream from the recommendation engine.
    """
    if is_recommendation_query(user_message):
        max_price, priority = extract_budget_and_priority(user_message)

        async for chunk in stream_recommendations(max_price, priority):
            yield chunk
        return

    messages = build_messages(chat_history, user_message)
    llm = create_llm()

    async for chunk in llm.astream(messages):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content


def generate_reply(chat_history, user_message):
    """
    Non-streaming chatbot response used by the current /chat endpoint.
    """
    if is_recommendation_query(user_message):
        max_price, priority = extract_budget_and_priority(user_message)

        rec_response = get_recommendations(
            max_price=max_price,
            priority=priority,
        )

        return rec_response["recommendations"]

    llm = create_llm()
    messages = build_messages(chat_history, user_message)
    response = llm.invoke(messages)
    return response.content
