import os
import sys
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
from olx_scraper_service import scrape_used_data  


load_dotenv()


MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
client = MongoClient(MONGO_URI)

DB_NAME = "MobileDB"
COLLECTION_NAME = "mobile_brands"
db = client[DB_NAME]
collection = db[COLLECTION_NAME]



def get_next_batch(models, start_index, batch_size=10):
    """
    Returns a batch of models in a safe, wrap-around manner.
    """
    end_index = start_index + batch_size

    if start_index >= len(models):  # wrap around fully
        start_index = 0
        end_index = batch_size

    batch = models[start_index:end_index]

    # If end_index exceeds list → wrap remaining items from start
    if not batch:
        batch = models[0:batch_size]

    return batch


def update_model_index(brand, current_index, models_len, batch_size=10):
    """
    Moves pointer forward and wraps around automatically.
    """
    new_index = current_index + batch_size
    if new_index >= models_len:
        new_index = 0

    collection.update_one(
        {"brand": brand},
        {"$set": {
            "model_index": new_index,
            "last_updated": datetime.utcnow()
        }}
    )

    return new_index


# ---------------------------
# MAIN ROUND-ROBIN SCRAPER
# ---------------------------

def run_round_robin_scraper(batch_size=10):
    print("======================================")
    print("Cron Job Started:", datetime.now())
    print("======================================")

    brands = list(collection.find({}))
    if not brands:
        print("❌ No brands found in DB.")
        sys.exit(0)

    for brand_doc in brands:
        brand = brand_doc["brand"]
        models = brand_doc["models"]
        model_index = brand_doc.get("model_index", 0)

        print(f"\n🔵 Brand: {brand}")
        print(f"➡️  Starting at model_index: {model_index}")

        # 1️⃣ Get next 10 models
        batch = get_next_batch(models, model_index, batch_size)
        print(f"📌 Models in this batch ({len(batch)}): {batch}")

        # 2️⃣ Scrape each model
        for model in batch:
            print(f"\n🚀 Scraping → {brand} / {model}")
            try:
                scrape_used_data(model, brand)
            except Exception as e:
                print(f"❌ Error scraping {brand} {model}:", e)

        # 3️⃣ Update index in DB
        update_model_index(brand, model_index, len(models), batch_size)

        print(f"✔️ Updated model_index → {brand}")

    print("\n======================================")
    print("Cron Job Finished:", datetime.now())
    print("======================================")




if __name__ == "__main__":
    run_round_robin_scraper(batch_size=10)
