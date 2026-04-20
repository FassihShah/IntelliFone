from bs4 import BeautifulSoup
import requests
from time import sleep
import random
from urllib.parse import quote_plus
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime, timezone
from bson import ObjectId
import os
import time
import json
import re

from models import UsedMobile
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


load_dotenv()

# Constants
BASE_URL = "https://www.olx.com.pk/mobile-phones_c1411"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Mongo Setup
MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
DB_NAME = "MobileDB"
COLLECTION_NAME = "used_mobiles"
BRAND_COLLECTION_NAME = "mobile_brands"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
brand_collection = db[BRAND_COLLECTION_NAME]

def ensure_olx_indexes():
    try:
        collection.create_index([("link", 1)], unique=True)

        for index_name, index_info in collection.index_information().items():
            if index_info.get("key") == [("extraction_date", 1)] and "expireAfterSeconds" in index_info:
                collection.drop_index(index_name)
                print("Dropped old TTL index on extraction_date:", index_name)

        collection.create_index([("extraction_date", 1)])
    except Exception as e:
        print("OLX index setup skipped/failed:", e)


# LLM Setup
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    temperature=0,
    max_tokens=1500
)


# ============================================================
# COMBINED MULTI-TASK PROMPT (Model Check + Extraction)
# ============================================================
combined_prompt = ChatPromptTemplate.from_template("""
You have TWO responsibilities in a SINGLE step:

============================================================================
### 1️⃣ STRICT MODEL & BRAND VERIFICATION
============================================================================

Verify if this listing exactly matches:

- Expected Brand: {brand}
- Expected Model: {model}

MATCHING RULES:
✔ Model must match exactly, ignoring only "4G" / "5G" suffix differences  
✔ Examples of accepted equivalence:
   - "Oppo Reno 13" == "Oppo Reno 13 5G"
   - "Redmi Note 13" == "Redmi Note 13 4G"

✔ DO NOT accept:
   - A52 vs A52s
   - Pixel 6 vs Pixel 6A
   - iPhone 13 vs iPhone 13 Pro

ADDITIONAL STRICT RULE:
Reject the listing only if it mentions MULTIPLE DIFFERENT PHONE MODELS.

Do NOT treat RAM, storage, or numbers (e.g., 6GB, 128GB, 6/128) as separate models.

Valid single-model examples if model is "Pixel 6A" (ACCEPT):
- "Google Pixel 6A official PTA approved 6gb 128gb"
- "PIXEL 6A OFFICIAL PTA"
- "Google Pixel 6A 6/128"

Invalid multi-model examples (REJECT):
- "Pixel 7a / 7 Pro / 7 / 6A / 8 / 8 Pro 9xl"
- "iPhone 11 / 12 / 13 all available"
- "Samsung A52 / A52s / A53 PTA approved"

A listing should only be rejected if it clearly lists **two or more different device models**,
not when it simply contains RAM or storage numbers.

If multiple distinct phone models appear → return "skip".

No JSON. No extra text.

============================================================================
### 2️⃣ IF MODEL MATCHES → RETURN STRUCTURED JSON
============================================================================

Return a JSON object with fields:

- brand  
- model  
- ram  
- storage  
- condition (tone-based 1–10)  
- condition_score → ALWAYS null
- pta_approved  
- is_panel_changed  
- screen_crack  
- panel_dot  
- panel_line  
- panel_shade
- camera_lens_ok  
- fingerprint_ok  
- with_box  
- with_charger  
- price  
- city  
- listing_source = "OLX"  
- images (list of URLs)  
- post_date  

============================================================================
### ASSUMPTION RULES
============================================================================

If NOT mentioned:
- is_panel_changed = false  
- screen_crack = false  
- panel_dot = false  
- panel_line = false  
- panel_shade = false
- camera_lens_ok = true  
- fingerprint_ok = true  
- with_box = false  
- with_charger = false  

PTA RULE:
- Default: true  
- false if text contains:
  "non PTA", "PTA not approved", "SIM lock", "JV phone"

Condition rating (1–10) based ONLY on tone.  
It must not be null.
Ignore technical issues in rating.

============================================================================
### OUTPUT RULES
============================================================================
❗ If mismatched → output ONLY: "skip"  
❗ If matched → output ONLY raw JSON (no markdown)  
                                                   
FORMATTING RULES (CRITICAL):
- "ram" MUST be a string. If it is a number like 6 or 8, convert to "6GB" or "8GB".
- "storage" MUST be a string. If it is a number like 128 or 256, convert to "128GB" or "256GB".
- "condition" MUST be an integer between 1 and 10. If your analysis gives a decimal like 8.5, round it to the nearest integer (8 or 9).

============================================================================

LISTING INPUT:
Title: {title}
Description: {description}
Free-text Condition: {condition}
Price: {price}
Location: {location}
Images: {images}

""")

combined_chain = combined_prompt | llm | StrOutputParser()


# ============================================================
# Rate Limit Handler
# ============================================================
last_llm_call = 0


SCRAPINGBEE_API_KEY = os.getenv("SCRAPINGBEE_API_KEY")

def direct_fetch(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        return response
    except Exception as e:
        print("Direct fetch failed:", e)
        return None


def fetch(url):
    """Proxy OLX requests through ScrapingBee to bypass blocking."""
    if not SCRAPINGBEE_API_KEY:
        print("ScrapingBee API key missing. Falling back to direct request.")
        return direct_fetch(url)

    try:
        response = requests.get(
            "https://app.scrapingbee.com/api/v1/",
            params={
                "api_key": SCRAPINGBEE_API_KEY,
                "url": url,
                "render_js": "false",
                "country_code": "pk",
                "block_resources": "true",
            },
            timeout=30,
        )
        response.raise_for_status()
        return response
    except Exception as e:
        print("ScrapingBee fetch failed:", e)
        print("Trying direct request instead:", url)
        return direct_fetch(url)



def rate_limit_pause():
    global last_llm_call
    now = time.time()
    elapsed = now - last_llm_call
    min_interval = 6

    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

    last_llm_call = time.time()




def sanitize_llm_json(raw_json: str):
    """
    Cleans LLM output before JSON parsing.
    Fixes:
    - Markdown code fences (```json ... ```)
    - Leading/trailing backticks
    - RAM returned as int/float
    - Storage returned as int/float
    - Condition returned as float/string
    - Price returned as string with currency symbols
    """

    # Remove any markdown code fences
    raw_json = raw_json.strip()
    if raw_json.startswith("```"):
        raw_json = raw_json.strip("`")
        raw_json = raw_json.replace("json", "", 1).strip()

    # Remove accidental triple or double backticks
    raw_json = raw_json.replace("```", "")
    raw_json = raw_json.replace("``", "")

    # 3Remove stray backticks anywhere
    raw_json = raw_json.replace("`", "")

    # Parse cleaned JSON
    try:
        data = json.loads(raw_json)
    except Exception as e:
        print("❌ Failed to parse JSON after cleaning:", raw_json)
        raise e

    # Fix RAM
    if isinstance(data.get("ram"), int):
        data["ram"] = f"{data['ram']}GB"
    elif isinstance(data.get("ram"), float):
        data["ram"] = f"{int(data['ram'])}GB"

    # Fix Storage
    if isinstance(data.get("storage"), int):
        data["storage"] = f"{data['storage']}GB"
    elif isinstance(data.get("storage"), float):
        data["storage"] = f"{int(data['storage'])}GB"


    # Fix Condition (1–10 integer)
    if isinstance(data.get("condition"), float):
        data["condition"] = round(data["condition"])
    elif isinstance(data.get("condition"), str):
        try:
            data["condition"] = round(float(data["condition"]))
        except:
            data["condition"] = None


    # Fix Price
    price = data.get("price")

    if isinstance(price, str):
        # Remove currency symbols, commas, text
        cleaned = re.sub(r"[^\d]", "", price)
        data["price"] = int(cleaned) if cleaned else None
    elif isinstance(price, float):
        data["price"] = int(price)
    elif isinstance(price, int):
        data["price"] = price
    else:
        data["price"] = None

    return json.dumps(data)




# ============================================================
# Save To Mongo
# ============================================================
def save_to_db(mobile: UsedMobile, link: str):
    collection = db[COLLECTION_NAME]
    now = datetime.now(timezone.utc)

    data = mobile.model_dump()
    data["extraction_date"] = now
    data["_id"] = ObjectId()
    data["link"] = link   # ✅ ADD LINK MANUALLY HERE

    try:
        collection.insert_one(data)
        print("✅ Saved new listing:", link)
        return True

    except Exception as e:
        if "duplicate key error" in str(e):
            print("⚠️ Duplicate listing — skipping:", link)
            return False

        print("❌ MongoDB Insert Error:", e)
        return False



# ============================================================
# Combined Extraction + Verification
# ============================================================
def extract_data(data: dict, model, brand):
    try:
        rate_limit_pause()

        llm_result = combined_chain.invoke({
            "title": data.get("title", ""),
            "description": data.get("description", ""),
            "brand": brand,
            "model": model,
            "condition": data.get("condition", ""),
            "price": data.get("price", ""),
            "location": data.get("location", ""),
            "images": data.get("images", "")
        }).strip()

        if llm_result == "skip":
            print("❌ Skipped (Model mismatch):", data.get("title", ""))
            return False

        sanitized = sanitize_llm_json(llm_result)
        mobile = UsedMobile.model_validate_json(sanitized)

        mobile.post_date = data.get("post_date", "")

        images = data.get("images", "")
        mobile.images = [img.strip() for img in images.split(",") if img.strip()]

        success = save_to_db(mobile, data["link"])

        if success:
            print(
                f"Saved as: {mobile.brand} {mobile.model} "
                f"| title: {data.get('title', '')}"
            )
            print(f"✅ Extracted: {mobile.model} with title: {data.get('title', '')}")
            return True

    except Exception as e:
        print("❌ LLM Extraction Failed:", e)
        return False


# ============================================================
# Scrape OLX Listings
# ============================================================
IGNORED_MODEL_TOKENS = {"4g", "5g", "lte"}
MODEL_VARIANT_TOKENS = {
    "edge",
    "fe",
    "flip",
    "fold",
    "lite",
    "max",
    "mini",
    "note",
    "plus",
    "pro",
    "se",
    "ultra",
}
OPTIONAL_MODEL_TOKENS_BY_BRAND = {
    "samsung": {"galaxy"},
}
ACCESSORY_TERMS = {
    "accessory",
    "accessories",
    "adapter",
    "airpods",
    "cable",
    "case",
    "charger",
    "charging",
    "cover",
    "handsfree",
    "protector",
}
REQUIRED_MODEL_CATALOG = None


def raw_tokens(text: str):
    normalized = (text or "").lower()
    normalized = normalized.replace("+", " plus ")
    normalized = re.sub(r"\bpromax\b", "pro max", normalized)
    return re.findall(r"[a-z0-9]+", normalized)


def compact_text(text: str):
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def model_required_tokens(model: str, brand: str):
    brand_key = (brand or "").lower()
    brand_tokens = set(raw_tokens(brand))
    optional_tokens = OPTIONAL_MODEL_TOKENS_BY_BRAND.get(brand_key, set())

    return [
        token
        for token in raw_tokens(model)
        if token not in brand_tokens
        and token not in IGNORED_MODEL_TOKENS
        and token not in optional_tokens
    ]


def load_required_model_catalog():
    """
    Builds a conservative list of known models from mobile_brands.
    We only opportunistically save off-query listings if they match this list.
    """
    global REQUIRED_MODEL_CATALOG

    if REQUIRED_MODEL_CATALOG is not None:
        return REQUIRED_MODEL_CATALOG

    catalog = []

    try:
        for brand_doc in brand_collection.find({}):
            brand = brand_doc.get("brand", "")
            models = brand_doc.get("models", [])

            for model in models:
                tokens = model_required_tokens(model, brand)
                if not tokens:
                    continue

                catalog.append({
                    "brand": brand,
                    "model": model,
                    "tokens": tokens,
                    "compact": "".join(tokens),
                    "score": (len(tokens), len("".join(tokens))),
                })

    except Exception as e:
        print("Required model catalog load failed:", e)
        catalog = []

    REQUIRED_MODEL_CATALOG = catalog
    return REQUIRED_MODEL_CATALOG


def title_contains_required_model(title: str, required_tokens):
    title_token_set = set(raw_tokens(title))
    compact_title = compact_text(title)
    compact_required = "".join(required_tokens)
    unmatched_variants = MODEL_VARIANT_TOKENS.intersection(title_token_set) - set(required_tokens)
    compact_variant_extensions = [
        variant
        for variant in MODEL_VARIANT_TOKENS - set(required_tokens)
        if f"{compact_required}{variant}" in compact_title
    ]

    if unmatched_variants or compact_variant_extensions:
        return False

    if all(token in title_token_set for token in required_tokens):
        return True

    return bool(compact_required and compact_required in compact_title)


def choose_best_model_candidate(candidates):
    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda item: item["score"],
        reverse=True,
    )

    best = candidates[0]
    tied = [
        candidate
        for candidate in candidates
        if candidate["score"] == best["score"]
        and (
            candidate["brand"] != best["brand"]
            or candidate["model"] != best["model"]
        )
    ]

    if tied:
        return None

    if len(candidates) > 1:
        second_best = candidates[1]
        if second_best["score"] == best["score"]:
            return None

    return best


def detect_required_model_from_title(title: str, fallback_model: str, fallback_brand: str):
    """
    Detects the exact required model represented by a listing title.
    This is intentionally limited to models from mobile_brands and remains
    backed by the LLM's strict verification after the detail page is fetched.
    """
    lower_title = (title or "").lower()
    accessory_hits = ACCESSORY_TERMS.intersection(set(raw_tokens(lower_title)))

    if accessory_hits and "phone" not in lower_title and "mobile" not in lower_title:
        return None

    fallback_tokens = model_required_tokens(fallback_model, fallback_brand)
    if fallback_tokens and title_contains_required_model(title, fallback_tokens):
        return fallback_brand, fallback_model

    catalog = load_required_model_catalog()
    candidates = [
        item
        for item in catalog
        if title_contains_required_model(title, item["tokens"])
    ]

    detected = choose_best_model_candidate(candidates)

    if detected:
        return detected["brand"], detected["model"]

    if candidates:
        return None

    return None


def get_ads_from_page(page_num, model_query, brand):

    if brand.lower() not in model_query.lower():
        full_query = f"{brand} {model_query}"
    else:
        full_query = model_query

    search_term = quote_plus(full_query).replace("+", "-")
    url = f"https://www.olx.com.pk/items/q-{search_term}?page={page_num}"
    print(f"Scraping Page URL: {url}")

    res = fetch(url)
    if not res:
        return [], 0

    soup = BeautifulSoup(res.text, "html.parser")

    ads = soup.select("li[aria-label='Listing']")
    ads_found = len(ads)
    listings = []

    for ad in ads:
        try:
            title_tag = ad.select_one("h2._1093b649")
            price_tag = ad.select_one("div[aria-label='Price'] span")
            location_tag = ad.select_one("span.f047db22")
            link_tag = ad.find("a", href=True)

            if not all([title_tag, price_tag, location_tag, link_tag]):
                continue

            title = title_tag.text.strip()
            price = price_tag.text.strip()
            location = location_tag.text.strip()
            link = "https://www.olx.com.pk" + link_tag["href"]

            detected_model = detect_required_model_from_title(title, model_query, brand)
            if not detected_model:
                print("Skipping detail fetch (title pre-filter):", title)
                continue

            target_brand, target_model = detected_model

            if target_brand != brand or target_model != model_query:
                print(
                    "Detected required model from off-query result:",
                    f"{target_brand} {target_model}",
                    "| title:",
                    title,
                )

            ad_res = fetch(link)
            if not ad_res:
                continue

            ad_soup = BeautifulSoup(ad_res.text, "html.parser")

            desc_tag = ad_soup.select_one("div[aria-label='Description'] div._7a99ad24 span")
            description = desc_tag.text.strip() if desc_tag else ""

            details = {}
            detail_tags = ad_soup.select("div[aria-label='Details'] div._0272c9dc.cd594ce1")

            for detail in detail_tags:
                spans = detail.find_all("span")
                if len(spans) == 2:
                    details[spans[0].text.strip()] = spans[1].text.strip()

            image_tags = ad_soup.select("div.image-gallery-slide img")
            image_urls = [img['src'] for img in image_tags if img.get('src')]

            data = {
                "title": title,
                "price": price,
                "location": location,
                "link": link,
                "description": description,
                "brand": details.get("Brand", ""),
                "model": details.get("Model", ""),
                "condition": details.get("Condition", ""),
                "images": ", ".join(image_urls)
            }

            success = extract_data(data, target_model, target_brand)
            if success:
                listings.append(data)

        except Exception as e:
            print("Skipping Ad, Error:", e)
            continue

    return listings, ads_found


# ============================================================
# Main Scraper Function
# ============================================================
def scrape_used_data(model: str, brand: str):
    print(f"🚀 Collecting data for model: {model}")

    count_saved = 0
    page_num = 1

    try:
        while True:
            listings_count = 0

            listings, ads_found = get_ads_from_page(page_num, model, brand)

            # We no longer store listings, we only count them
            for _ in listings:
                listings_count += 1
                count_saved += 1

            if ads_found == 0:
                print(f"No more listings on page {page_num}. Stopping.")
                break

            if count_saved >= 150:
                print("Reached limit of 150 successful extractions. Stopping.")
                break

            page_num += 1
            sleep(random.uniform(3, 6))

    except Exception as e:
        print("❌ Error while scraping data:", e)

    print(f"📦 Total listings saved to DB: {count_saved}")


# ============================================================
# TEST RUN
# ============================================================
# scrape_used_data("Pixel 6A", "Google")
