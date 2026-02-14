from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import re
from RecommendationEngine.recommendation_service import get_recommendations
from RecommendationEngine import recommendation_service

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

def extract_budget_and_priority(message: str):
    """
    Extract budget (number) and priority from user message.
    Defaults are applied if missing.
    """

    # Extract budget (e.g. 70000, 70k, 70,000)
    budget_match = re.search(r'(\d{2,6})', message.replace(",", ""))
    max_price = float(budget_match.group(1)) if budget_match else 70000

    # Priority keywords
    priorities = {
        "gaming": ["gaming", "performance", "fps"],
        "camera": ["camera", "photography", "selfie"],
        "battery": ["battery", "backup", "mah"],
        "general": ["all round", "balanced", "daily use"]
    }

    priority = "general"

    for key, words in priorities.items():
        if any(word in message.lower() for word in words):
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
        "purchase"
    ]

    message = message.lower()
    return any(keyword in message for keyword in keywords)



def generate_reply(chat_history, user_message):
    # 🔀 Step 1: check if recommendation intent
    if is_recommendation_query(user_message):
        max_price, priority = extract_budget_and_priority(user_message)

        rec_response = get_recommendations(
            max_price=max_price,
            priority=priority
        )

        return rec_response["recommendations"]

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_message))

    response = llm.invoke(messages)
    return response.content

