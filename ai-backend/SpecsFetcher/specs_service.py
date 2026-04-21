import os
import re
from datetime import datetime, timezone
from typing import Optional

import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient

from models import NewMobile


MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
DB_NAME = "MobileDB"
SPECS_COLLECTION_NAME = "mobile_specs"

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
SERPAPI_SEARCH_URL = "https://serpapi.com/search"
GSMARENA_RESULT_PATTERN = re.compile(r"-\d+\.php(?:$|\?)")

mongo_client = MongoClient(MONGO_URI) if MONGO_URI else None
specs_collection = (
    mongo_client[DB_NAME][SPECS_COLLECTION_NAME]
    if mongo_client is not None
    else None
)


def ensure_specs_cache_indexes():
    if specs_collection is None:
        return

    specs_collection.create_index([("normalized_key", 1)], unique=True)
    specs_collection.create_index([("brand", 1), ("model", 1)])
    specs_collection.create_index([("updated_at", -1)])


def normalize_value(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", (value or "").lower())
    return re.sub(r"\s+", " ", normalized).strip()


def normalized_specs_key(brand: str, model: str) -> str:
    return f"{normalize_value(brand)}|{normalize_value(model)}"


def convert_specs_to_mobile(specs: dict) -> NewMobile:
    def get(*keys: str) -> Optional[str]:
        for key in keys:
            value = specs.get(key)
            if value:
                return value
        return None

    def extract_year_from_release(release_info: Optional[str]) -> Optional[int]:
        if release_info:
            match = re.search(r"\b(19|20)\d{2}\b", release_info)
            if match:
                return int(match.group(0))
        return None

    def parse_ram_and_storage(mixed_string: str):
        ram_set = set()
        storage_set = set()

        matches = re.findall(
            r"(\d+(?:GB|TB))\s+(\d+(?:GB|MB))\s+RAM",
            mixed_string,
            flags=re.IGNORECASE,
        )

        for storage, ram in matches:
            storage_set.add(storage.upper())
            ram_set.add(ram.upper())

        return sorted(storage_set), sorted(ram_set)

    mobile = NewMobile(
        brand=None,
        model=None,
        os=get("Platform - OS"),
        release_year=extract_year_from_release(get("Launch - Announced")),
        screen_size=get("Display - Size"),
        screen_resolution=get("Display - Resolution"),
        battery_capacity=get("Battery - Type"),
        main_camera=get(
            "Main Camera - Single",
            "Main Camera - Dual",
            "Main Camera - Triple",
            "Main Camera - Quad",
        ),
        selfie_camera=get(
            "Selfie camera - Single",
            "Selfie camera - Dual",
        ),
        chipset=get("Platform - Chipset"),
        cpu=get("Platform - CPU"),
        gpu=get("Platform - GPU"),
        network=get("Network - Technology"),
        network_bands=get("Network - 2G bands"),
        sim=get("Network - SIM"),
        weight=get("Body - Weight"),
        dimensions=get("Body - Dimensions"),
        usb=get("Comms - USB"),
        sensors=get("Features - Sensors"),
        price=get("Misc - Price"),
    )

    memory_specs = specs.get("Memory - Internal")
    if memory_specs:
        storage_list, ram_list = parse_ram_and_storage(memory_specs)
        mobile.storage = ", ".join(storage_list) if storage_list else None
        mobile.ram = ", ".join(ram_list) if ram_list else None

    return mobile


def search_gsmarena_url(brand: str, model: str) -> Optional[str]:
    if not SERPAPI_API_KEY:
        raise RuntimeError("Missing SERPAPI_API_KEY in environment")

    query = f"{brand} {model} site:gsmarena.com"
    response = requests.get(
        SERPAPI_SEARCH_URL,
        params={
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "hl": "en",
            "google_domain": "google.com",
        },
        timeout=20,
    )
    response.raise_for_status()

    data = response.json()
    for item in data.get("organic_results", []):
        href = item.get("link", "")
        if "gsmarena.com" in href and GSMARENA_RESULT_PATTERN.search(href):
            return href

    return None


def scrape_gsmarena_specs(gsmarena_url: str, brand: str, model: str) -> NewMobile:
    response = requests.get(
        gsmarena_url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    specs = {}
    specs_list = soup.find("div", id="specs-list")
    current_section = None

    if specs_list:
        for row in specs_list.select("tr"):
            section_header = row.find("th", {"scope": "row"})
            if section_header:
                current_section = section_header.text.strip()

            key_td = row.find("td", class_="ttl")
            value_td = row.find("td", class_="nfo")

            if key_td and value_td and current_section:
                key = key_td.text.strip()
                value = value_td.text.strip()
                specs[f"{current_section} - {key}"] = value

    mobile = convert_specs_to_mobile(specs)
    mobile.brand = brand
    mobile.model = model
    return mobile


def read_specs_cache(brand: str, model: str):
    if specs_collection is None:
        return None

    normalized_key = normalized_specs_key(brand, model)
    doc = specs_collection.find_one({"normalized_key": normalized_key})
    if not doc:
        return None

    specs = NewMobile.model_validate(doc["specs"])
    return {
        "brand": doc["brand"],
        "model": doc["model"],
        "gsmarena_url": doc.get("gsmarena_url"),
        "specs": specs,
        "cached": True,
        "updated_at": doc.get("updated_at"),
    }


def save_specs_cache(brand: str, model: str, gsmarena_url: str, specs: NewMobile):
    if specs_collection is None:
        return None

    now = datetime.now(timezone.utc)
    normalized_key = normalized_specs_key(brand, model)
    specs_collection.update_one(
        {"normalized_key": normalized_key},
        {
            "$set": {
                "normalized_key": normalized_key,
                "brand": brand,
                "model": model,
                "gsmarena_url": gsmarena_url,
                "specs": specs.model_dump(),
                "source": "gsmarena",
                "updated_at": now,
            }
        },
        upsert=True,
    )
    return now


def fetch_mobile_specs(brand: str, model: str, refresh: bool = False):
    brand = brand.strip()
    model = model.strip()

    if not refresh:
        cached = read_specs_cache(brand, model)
        if cached:
            return cached

    gsmarena_url = search_gsmarena_url(brand, model)
    if not gsmarena_url:
        raise LookupError(f"No GSMArena result found for {brand} {model}")

    specs = scrape_gsmarena_specs(gsmarena_url, brand, model)
    updated_at = save_specs_cache(brand, model, gsmarena_url, specs)

    return {
        "brand": brand,
        "model": model,
        "gsmarena_url": gsmarena_url,
        "specs": specs,
        "cached": False,
        "updated_at": updated_at,
    }



print(fetch_mobile_specs("Samsung", "Galaxy S21"))
