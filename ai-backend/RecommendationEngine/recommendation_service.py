# recommendation_service.py

import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from pymongo import MongoClient


load_dotenv()

MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
client = MongoClient(MONGO_URI)
db = client["MobileDB"]
recommended_collection = db["phones"]


def ensure_recommendation_indexes():
    try:
        recommended_collection.create_index([("price_range", 1)])
        recommended_collection.create_index([("source_channel", 1)])
        recommended_collection.create_index([("source_weight", -1)])
    except Exception as e:
        print("Recommendation index setup skipped/failed:", e)


model = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    temperature=0.3,
)


class PhoneRecommendationInput(BaseModel):
    max_price: float = Field(description="Maximum price budget for the phone")
    priority: str = Field(description="User's priority (e.g., gaming performance, camera, battery life)")


def fetch_candidate_phones(max_price: float):
    """
    Fetch phones under the user's budget first.
    If none exist, fall back to nearby phones around the budget.
    """
    phones = list(recommended_collection.find({
        "price_range": {
            "$ne": None,
            "$lte": max_price,
        }
    }).sort("price_range", -1).limit(25))

    if not phones:
        phones = list(recommended_collection.find({
            "price_range": {
                "$ne": None,
                "$gte": max_price - 10000,
                "$lte": max_price + 5000,
            }
        }).sort("price_range", 1).limit(25))

    return phones


def build_prompt(max_price: float, priority: str, phones: list) -> str:
    candidates = []

    for idx, phone in enumerate(phones, 1):
        phone_name = phone.get("phone_name", "Unknown Phone")
        desc = phone.get("description", "No description available")
        source_channel = phone.get("source_channel") or "Unknown channel"
        source_weight = phone.get("source_weight", 1.0)
        price_range = phone.get("price_range")
        price_str = str(price_range) if price_range else "Price not available"
        desc = f"{desc} Source: {source_channel}. Source weight: {source_weight}"
        candidates.append(f"{idx}. {phone_name} - {desc} - {price_str}")

    return f"""
The user wants a phone with priority: {priority}.
Their budget is around {max_price}.

Here are some candidate phones:
{chr(10).join(candidates)}

Instructions:
1. Rank these phones based on how well they match the user's priority.
2. For each ranked phone, explain why it is a good (or not so good) match.
3. If no phone exactly matches the priority, recommend phones with generally good specs and justify why they are still strong alternatives.
4. Provide the final ranked list in a clear, user-friendly format.
Always use the currency Rs instead of writing any other currency symbol.

Format prices like this: Rs 70,000, Rs 80,000 (with commas).

use -> for headings.

Use **bold text** for phone names, prices, and key points.

Use *italic text* only for emphasis, not headings.

Present lists  numbered lists where appropriate.

Ensure consistent spacing and clean line breaks between sections.

Do not use * .

Avoid emojis; keep the tone professional and informative.

Keep explanations concise and structured in short paragraphs.
"""


async def stream_recommendations(max_price: float, priority: str):
    """
    Stream recommendation text chunk by chunk using DeepSeek.
    """
    phones = fetch_candidate_phones(max_price)

    if not phones:
        yield "No phones found in this price range."
        return

    prompt = build_prompt(max_price, priority, phones)

    async for chunk in model.astream(prompt):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content


def get_recommendations(max_price: float, priority: str) -> Dict[str, Any]:
    """
    Standard non-streaming recommendation response.
    """
    phones = fetch_candidate_phones(max_price)

    if not phones:
        return {"recommendations": "No phones found in this price range."}

    prompt = build_prompt(max_price, priority, phones)
    response = model.invoke(prompt)

    return {"recommendations": response.content}
