from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from pydantic import BaseModel, Field


load_dotenv()

MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
client = MongoClient(MONGO_URI)
db = client["MobileDB"]
recommended_collection = db["phones"]


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)


class PhoneRecommendationInput(BaseModel):
    max_price: float = Field(description="Maximum price budget for the phone")
    priority: str = Field(description="User's priority (e.g., gaming performance, camera, battery life)")



def get_recommendations(max_price: float, priority: str):
    """    Recommend phones under a price limit based on user priority.
    """
    phones = list(recommended_collection.find({
        "price_range": {"$lte": max_price + 5000},
        "price_range": {"$gte": max_price - 5000}
    }))

    if not phones:
        return {"recommendations": "No phones found in this price range."}

    candidates = []
    for idx, phone in enumerate(phones, 1):
        phone_name = phone.get("phone_name", "Unknown Phone")
        desc = phone.get("description", "No description available")
        price_range = phone.get("price_range", {})
        price_str = str(price_range) if price_range else "Price not available"
        candidates.append(f"{idx}. {phone_name} – {desc} – {price_str}")

    prompt = f"""
The user wants a phone with priority: {priority}.
Their budget is around {max_price}.

Here are some candidate phones:
{chr(10).join(candidates)}

Instructions:
1. Rank these phones based on how well they match the user's priority.
2. For each ranked phone, explain why it is a good (or not so good) match.
3. If no phone exactly matches the priority, recommend phones with generally good specs and justify why they are still strong alternatives.
4. Provide the final ranked list in a clear, user-friendly format.
"""


    response = model.invoke(prompt)
    return {"recommendations": response.text}