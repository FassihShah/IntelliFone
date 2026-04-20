import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from pymongo import MongoClient

from olx_scraper_service import ensure_olx_indexes, scrape_used_data


load_dotenv()


MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
client = MongoClient(MONGO_URI)

DB_NAME = "MobileDB"
COLLECTION_NAME = "mobile_brands"
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


def get_next_batch(models, start_index, batch_size=5):
    """
    Returns the next models in a safe, wrap-around manner.
    """
    if not models:
        return []

    start_index = start_index % len(models)
    batch_count = min(batch_size, len(models))

    return [
        models[(start_index + offset) % len(models)]
        for offset in range(batch_count)
    ]


def update_model_index(brand, new_index, models_len):
    """
    Saves the next model index and wraps around automatically.
    """
    if models_len <= 0:
        new_index = 0
    else:
        new_index = new_index % models_len

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

def run_round_robin_scraper(batch_size=5):
    ensure_olx_indexes()

    print("======================================")
    print("Cron Job Started:", datetime.now())
    print("======================================")

    brands = list(collection.find({}))
    if not brands:
        print("No brands found in DB.")
        sys.exit(0)

    for brand_doc in brands:
        brand = brand_doc["brand"]
        models = brand_doc["models"]

        if not models:
            print(f"No models found for brand: {brand}")
            continue

        model_index = brand_doc.get("model_index", 0) % len(models)

        print(f"\nBrand: {brand}")
        print(f"Starting at model_index: {model_index}")

        batch = get_next_batch(models, model_index, batch_size)
        print(f"Models in this batch ({len(batch)}): {batch}")

        for offset, model in enumerate(batch):
            current_model_index = (model_index + offset) % len(models)
            next_model_index = (current_model_index + 1) % len(models)

            print(f"\nScraping -> {brand} / {model}")
            try:
                scrape_used_data(model, brand)
            except Exception as e:
                print(f"Error scraping {brand} {model}:", e)

            saved_index = update_model_index(brand, next_model_index, len(models))
            print(f"Updated model_index -> {brand}: {saved_index}")

    print("\n======================================")
    print("Cron Job Finished:", datetime.now())
    print("======================================")


if __name__ == "__main__":
    run_round_robin_scraper(batch_size=5)
