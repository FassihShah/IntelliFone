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
STATE_COLLECTION_NAME = "cron_state"
STATE_DOC_ID = "olx_round_robin"
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
state_collection = db[STATE_COLLECTION_NAME]


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


def get_brand_cursor() -> int:
    state_doc = state_collection.find_one({"_id": STATE_DOC_ID}) or {}
    return int(state_doc.get("brand_index", 0))


def update_brand_cursor(new_index: int):
    state_collection.update_one(
        {"_id": STATE_DOC_ID},
        {"$set": {
            "brand_index": new_index,
            "last_updated": datetime.utcnow()
        }},
        upsert=True
    )


def get_brand_batch(brands, start_index: int, brands_per_run: int):
    if not brands:
        return [], start_index

    start_index = start_index % len(brands)
    batch_count = min(max(brands_per_run, 1), len(brands))

    selected = [
        brands[(start_index + offset) % len(brands)]
        for offset in range(batch_count)
    ]
    next_index = (start_index + batch_count) % len(brands)
    return selected, next_index


# ---------------------------
# MAIN ROUND-ROBIN SCRAPER
# ---------------------------

def run_round_robin_scraper(batch_size=5, brands_per_run=None):
    ensure_olx_indexes()

    print("======================================")
    print("Cron Job Started:", datetime.now())
    print("======================================")

    brands = list(collection.find({}).sort("brand", 1))
    if not brands:
        print("No brands found in DB.")
        sys.exit(0)

    if brands_per_run is None:
        brands_per_run = int(os.getenv("OLX_BRANDS_PER_RUN", "3"))

    start_brand_index = get_brand_cursor()
    selected_brands, next_brand_index = get_brand_batch(brands, start_brand_index, brands_per_run)

    print(f"Starting at brand_index: {start_brand_index}")
    print(f"Brands in this run ({len(selected_brands)}): {[brand_doc['brand'] for brand_doc in selected_brands]}")

    for brand_doc in selected_brands:
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

    update_brand_cursor(next_brand_index)
    print(f"\nUpdated global brand_index -> {next_brand_index}")

    print("\n======================================")
    print("Cron Job Finished:", datetime.now())
    print("======================================")


if __name__ == "__main__":
    run_round_robin_scraper(batch_size=5)
