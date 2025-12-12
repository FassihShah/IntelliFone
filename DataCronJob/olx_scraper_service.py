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

from models import UsedMobile
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Constants
BASE_URL = "https://www.olx.com.pk/mobile-phones_c1411"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Mongo Setup
MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
DB_NAME = "MobileDB"
COLLECTION_NAME = "used_mobiles"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

db[COLLECTION_NAME].create_index([("link", 1)], unique=True)   # Ensure link uniqueness, no duplicates
db[COLLECTION_NAME].create_index([("extraction_date", 1)], expireAfterSeconds=5184000)   # 60 days TTL 


# LLM Setup
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

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
Ignore technical issues in rating.

If unsure → use null.

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
last_gemini_call = 0


SCRAPINGBEE_API_KEY = os.getenv("SCRAPINGBEE_API_KEY")

def fetch(url):
    """Proxy OLX requests through ScrapingBee to bypass blocking."""
    api_url = (
        f"https://app.scrapingbee.com/api/v1/"
        f"?api_key={SCRAPINGBEE_API_KEY}"
        f"&render_js=false"
        f"&url={url}"
    )
    try:
        response = requests.get(api_url, timeout=30)
        return response
    except Exception as e:
        print("ScrapingBee fetch failed:", e)
        return None



def rate_limit_pause():
    global last_gemini_call
    now = time.time()
    elapsed = now - last_gemini_call
    min_interval = 6  # Gemini calls every 6 seconds

    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

    last_gemini_call = time.time()



def sanitize_llm_json(raw_json: str):
    """
    Cleans LLM output before JSON parsing.
    Fixes:
    - Markdown code fences (```json ... ```)
    - Leading/trailing backticks
    - RAM returned as int
    - Storage returned as int
    - Condition returned as float
    """

    # 1️⃣ Remove any markdown code fences
    raw_json = raw_json.strip()
    if raw_json.startswith("```"):
        # Remove ```json or ``` and ending ```
        raw_json = raw_json.strip("`")
        raw_json = raw_json.replace("json", "", 1).strip()

    # 2️⃣ Remove accidental triple or double backticks
    raw_json = raw_json.replace("```", "")
    raw_json = raw_json.replace("``", "")

    # 3️⃣ Remove stray backticks anywhere
    raw_json = raw_json.replace("`", "")

    # 4️⃣ Parse cleaned JSON
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

    # Fix Condition (integer only)
    if isinstance(data.get("condition"), float):
        data["condition"] = round(data["condition"])
    elif isinstance(data.get("condition"), str):
        # If LLM outputs "8.5" as string
        try:
            data["condition"] = round(float(data["condition"]))
        except:
            data["condition"] = None

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
            print(f"✅ Extracted: {mobile.model} with title: {data.get('title', '')}")
            return True

    except Exception as e:
        print("❌ LLM Extraction Failed:", e)
        return False


# ============================================================
# Scrape OLX Listings
# ============================================================
def get_ads_from_page(page_num, model_query, brand):

    if brand.lower() not in model_query.lower():
        full_query = f"{brand} {model_query}"
    else:
        full_query = model_query

    search_term = full_query.replace(" ", "-")
    url = f"https://www.olx.com.pk/items/q-{search_term}?page={page_num}"
    print(f"Scraping Page URL: {url}")

    scraper = requests.Session()
    scraper.headers.update(HEADERS)
    res = fetch(url)
    soup = BeautifulSoup(res.text, "html.parser")

    ads = soup.select("li[aria-label='Listing']")
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

            ad_res = fetch(link)
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

            success = extract_data(data, model_query, brand)
            if success:
                listings.append(data)

        except Exception as e:
            print("Skipping Ad, Error:", e)
            continue

    return listings


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

            listings = get_ads_from_page(page_num, model, brand)

            # We no longer store listings, we only count them
            for _ in listings:
                listings_count += 1
                count_saved += 1

            if listings_count == 0:
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
scrape_used_data("Galaxy A71", "Samsung")
