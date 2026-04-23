from typing import List
from pymongo.collection import Collection
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import re
import os
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

from models import UsedMobile

# =====================================================
# DB SETUP
# =====================================================
MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
DB_NAME = "MobileDB"
COLLECTION_NAME = "used_mobiles"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def ensure_price_prediction_indexes():
    try:
        for index_name, index_info in collection.index_information().items():
            if index_info.get("key") == [("extraction_date", 1)] and "expireAfterSeconds" in index_info:
                collection.drop_index(index_name)

        collection.create_index([("extraction_date", 1)])
    except Exception as e:
        print("Price prediction index setup skipped/failed:", e)

RECENT_LISTINGS_DAYS = 60
MIN_RECENT_LISTINGS = 50
MIN_TRAINING_LISTINGS = 20


def normalize_phone_text(value: str | None, brand: str | None = None) -> str:
    if not value:
        return ""

    text = value.lower()
    if brand:
        for token in re.findall(r"[a-z0-9]+", brand.lower()):
            text = re.sub(rf"\b{re.escape(token)}\b", " ", text)

    return "".join(re.findall(r"[a-z0-9]+", text))

# =====================================================
# CONDITION SCORE DERIVATION (FOR OLX DATA)
# =====================================================
def derive_condition_score(mobile: UsedMobile) -> float | None:
    if mobile.condition is None:
        return None

    score = mobile.condition * 2

    if mobile.screen_crack:
        score -= 5
    if mobile.panel_line:
        score -= 4
    if mobile.panel_dot:
        score -= 3
    if mobile.panel_shade:
        score -= 3
    if mobile.is_panel_changed:
        score -= 4

    return max(0, min(20, score))


# =====================================================
# FETCH TRAINING DATA
# =====================================================
def _build_valid_mobiles(docs) -> List[UsedMobile]:
    mobiles = []
    for doc in docs:
        try:
            if "images" in doc and isinstance(doc["images"], str):
                doc["images"] = [i.strip() for i in doc["images"].split(",") if i.strip()]

            mobile = UsedMobile(**doc)

            if mobile.condition_score is None:
                mobile.condition_score = derive_condition_score(mobile)

            if mobile.condition_score is not None:
                mobiles.append(mobile)

        except Exception:
            continue

    return mobiles


def _is_recent_listing(doc: dict, cutoff: datetime) -> bool:
    extraction_date = doc.get("extraction_date")
    if not isinstance(extraction_date, datetime):
        return False

    if extraction_date.tzinfo is not None:
        extraction_date = extraction_date.astimezone(timezone.utc).replace(tzinfo=None)

    return extraction_date >= cutoff


def _matches_requested_phone(doc: dict, input_model: str, input_brand: str | None = None) -> bool:
    expected_model = normalize_phone_text(input_model, input_brand)
    doc_model = normalize_phone_text(doc.get("model"), input_brand)

    if not expected_model or doc_model != expected_model:
        return False

    if input_brand:
        expected_brand = normalize_phone_text(input_brand)
        doc_brand = normalize_phone_text(doc.get("brand"))
        if doc_brand and doc_brand != expected_brand:
            return False

    return True


def fetch_training_data(input_model: str, input_brand: str | None = None, db: Collection = collection) -> List[UsedMobile]:
    query = {"model": {"$regex": re.escape(input_model), "$options": "i"}}
    cutoff = datetime.utcnow() - timedelta(days=RECENT_LISTINGS_DAYS)

    all_docs = [
        doc for doc in db.find(query).sort("extraction_date", -1)
        if _matches_requested_phone(doc, input_model, input_brand)
    ]
    recent_docs = [doc for doc in all_docs if _is_recent_listing(doc, cutoff)]

    recent_mobiles = _build_valid_mobiles(recent_docs)
    if len(recent_mobiles) >= MIN_RECENT_LISTINGS:
        return recent_mobiles

    mobiles = _build_valid_mobiles(all_docs)

    if len(mobiles) < MIN_TRAINING_LISTINGS:
        raise RuntimeError(f"Only {len(mobiles)} valid records found.")

    return mobiles


# =====================================================
# PREPROCESS INPUT
# =====================================================
def preprocess_input_mobile(input_mobile: UsedMobile) -> pd.DataFrame:
    row = input_mobile.model_dump()

    for field in ["ram", "storage"]:
        if isinstance(row.get(field), str):
            m = re.search(r"\d+", row[field])
            row[field] = int(m.group()) if m else None

    for k, v in row.items():
        if isinstance(v, bool):
            row[k] = int(v)

    df = pd.DataFrame([row])
    df.drop(columns=["price", "images", "post_date", "listing_source", "city"],
            inplace=True, errors="ignore")

    return df


# =====================================================
# PREPROCESS TRAINING DATA
# =====================================================
def preprocess_training_data(training_data: List[UsedMobile]) -> pd.DataFrame:
    rows = []

    for mobile in training_data:
        row = mobile.model_dump()

        for field in ["ram", "storage"]:
            if isinstance(row.get(field), str):
                m = re.search(r"\d+", row[field])
                row[field] = int(m.group()) if m else 6

        for k, v in row.items():
            if isinstance(v, bool):
                row[k] = int(v)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.drop(columns=["images", "post_date", "listing_source", "city", "model", "brand"],
            inplace=True, errors="ignore")

    return df


# =====================================================
# TRAIN MODEL
# =====================================================
def train_model(training_df: pd.DataFrame) -> RandomForestRegressor:
    df = training_df.dropna(subset=["price", "condition_score"])

    X = df.drop(columns=["price"])
    y = df["price"]

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=18,
        random_state=42
    )

    model.fit(X, y)
    return model


# =====================================================
# MARKET UNCERTAINTY (NEW – DATA DRIVEN)
# =====================================================
def compute_market_uncertainty(training_df: pd.DataFrame) -> float:
    prices = training_df["price"]

    if len(prices) < 30:
        return 0.05  # statistical fallback

    q1 = prices.quantile(0.25)
    q3 = prices.quantile(0.75)
    iqr = q3 - q1
    median = prices.median()

    if median == 0:
        return 0.05

    uncertainty = iqr / median

    return float(min(max(uncertainty, 0.02), 0.15))


def compute_dynamic_price_range(base_price: float, uncertainty: float):
    delta = base_price * uncertainty

    # Keep consumer-facing ranges tighter, especially for sub-100k phones.
    if base_price < 100000:
        max_total_spread = 10000
    elif base_price < 150000:
        max_total_spread = 15000
    elif base_price < 250000:
        max_total_spread = 20000
    else:
        max_total_spread = 30000

    delta = min(delta, max_total_spread / 2)

    min_price = round((base_price - delta) / 500) * 500
    max_price = round((base_price + delta) / 500) * 500

    return int(min_price), int(max_price)


# =====================================================
# PRICE PREDICTION
# =====================================================
def predict_price_range(model, input_df, training_df, mobile, ai_flags):
    df = input_df.copy()
    df.drop(columns=["model", "brand"], inplace=True, errors="ignore")

    # Base ML prediction
    base_price = model.predict(df)[0]

    # Condition score influence
    base_price *= (0.7 + 0.015 * mobile.condition_score)

    # AI-based verification penalties
    if mobile.screen_crack and not ai_flags.get("screen_crack", False):
        base_price *= 0.7

    if mobile.panel_dot and not ai_flags.get("panel_dot", False):
        base_price *= 0.75

    if mobile.panel_line and not ai_flags.get("panel_line", False):
        base_price *= 0.7

    # AI cannot detect these reliably
    if mobile.panel_shade:
        base_price *= 0.75

    if mobile.is_panel_changed:
        base_price *= 0.8

    if not mobile.camera_lens_ok:
        base_price *= 0.9

    if not mobile.fingerprint_ok:
        base_price *= 0.85

    if not mobile.pta_approved:
        base_price *= 0.8

    # Market-driven price range 
    uncertainty = compute_market_uncertainty(training_df)
    min_price, max_price = compute_dynamic_price_range(base_price, uncertainty)

    return {
        "min_price": int(min_price),
        "max_price": int(max_price)
    }



# =====================================================
# FINAL PIPELINE
# =====================================================
def run_pipeline(input_mobile: UsedMobile, ai_flags: dict, db: Collection = collection):
    training_data = fetch_training_data(input_mobile.model, input_mobile.brand, db)

    input_df = preprocess_input_mobile(input_mobile)
    training_df = preprocess_training_data(training_data)

    model = train_model(training_df)

    return predict_price_range(model, input_df, training_df, input_mobile, ai_flags)




ai_flags = {
    "screen_crack": False,
    "panel_dot": False,
    "panel_line": False
}

input_mobile = UsedMobile(
    brand="Samsung",
    model="Galaxy A71",
    ram="8GB",
    storage="128GB",
    condition_score=15.5,
    is_panel_changed=False,
    screen_crack=False,
    panel_dot=False,
    panel_line=True,
    panel_shade=False,
    camera_lens_ok=True,
    fingerprint_ok=True,
    pta_approved=True,
    price=None
)

# result = run_pipeline(input_mobile, ai_flags)
# print(result)
