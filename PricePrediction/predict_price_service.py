from typing import List
from datetime import datetime, timedelta, timezone
from pymongo.collection import Collection
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import re
import os
from pymongo import MongoClient
from dotenv import load_dotenv


load_dotenv()


from models import UsedMobile

MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")

DB_NAME = "MobileDB"
COLLECTION_NAME = "used_mobiles"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

MAX_DATA_AGE_DAYS = 30



def fetch_training_data(input_model: str, db: Collection = collection) -> List[UsedMobile]:
    """Fetch recent training data from MongoDB (OLX listings)."""
    
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=MAX_DATA_AGE_DAYS)

    query = {
        "model": {"$regex": re.escape(input_model), "$options": "i"},
        "extraction_date": {"$gte": cutoff_date}
    }

    training_data = []
    result = db.find(query)

    for doc in result:
        try:
            # Normalize image field
            if "images" in doc and isinstance(doc["images"], str):
                doc["images"] = [img.strip() for img in doc["images"].split(",") if img.strip()]
            
            training_data.append(UsedMobile(**doc))
        except Exception as e:
            print("Skipping record:", e)

    if len(training_data) < 150:
        raise RuntimeError(f"⚠️ Only {len(training_data)} fresh records found. Need 150 minimum.")

    return training_data



def preprocess_input_mobile(input_mobile: UsedMobile) -> pd.DataFrame:
    """Preprocess the user's mobile input for ML prediction."""
    
    input_dict = input_mobile.model_dump()

    # Convert ram/storage from "8GB" → 8
    for field in ["ram", "storage"]:
        val = input_dict.get(field)
        if isinstance(val, str) and "GB" in val.upper():
            num = ''.join(filter(str.isdigit, val))
            input_dict[field] = int(num) if num else None

    # Convert bool → 0/1
    for key, value in input_dict.items():
        if isinstance(value, bool):
            input_dict[key] = int(value)

    df = pd.DataFrame([input_dict])

    # Columns not used by ML
    drop_cols = ["price", "images", "post_date", "listing_source", "city"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    return df



def preprocess_training_data(training_data: List[UsedMobile]) -> pd.DataFrame:
    """Convert OLX scraped training data into ML-friendly dataframe."""
    
    processed = []
    fallback_ram = fallback_storage = None

    # Find any valid RAM/storage to use as fallback
    for item in training_data:
        if item.ram and item.storage and "GB" in item.ram and "GB" in item.storage:
            fallback_ram = item.ram
            fallback_storage = item.storage
            break

    if not fallback_ram or not fallback_storage:
        raise ValueError("No fallback RAM/storage found.")

    for item in training_data:
        row = item.model_dump()

        # Ensure RAM/Storage exist
        row["ram"] = row.get("ram") or fallback_ram
        row["storage"] = row.get("storage") or fallback_storage

        # Convert values like "8GB" → 8
        for field in ["ram", "storage"]:
            val = row[field]
            if isinstance(val, str):
                match = re.search(r'\d+', val)
                row[field] = int(match.group()) if match else 6

        # Convert booleans to numeric
        for key, value in row.items():
            if isinstance(value, bool):
                row[key] = int(value)

        processed.append(row)

    df = pd.DataFrame(processed)

    # Remove columns not used by ML
    df.drop(columns=["images", "post_date", "listing_source", "city", "model", "brand"],
            inplace=True, errors="ignore")

    return df



def merge_ai_user_flags(ai_flags: dict, mobile: UsedMobile):
    """
    AI detection overrides user input.
    User's flags apply only if AI did NOT detect the damage.
    """

    return {
        "screen_crack": ai_flags.get("screen_crack", False) or mobile.screen_crack,
        "panel_dot": ai_flags.get("panel_dot", False) or mobile.panel_dot,
        "panel_line": ai_flags.get("panel_line", False) or mobile.panel_line,
        "panel_shade": mobile.panel_shade,  # AI cannot detect shade
        "is_panel_changed": mobile.is_panel_changed,
        "camera_lens_ok": mobile.camera_lens_ok,
        "fingerprint_ok": mobile.fingerprint_ok,
        "pta_approved": mobile.pta_approved
    }



def train_model(training_df: pd.DataFrame) -> RandomForestRegressor:
    """Train the RandomForest price prediction model."""
    
    df = training_df.dropna(subset=["price"])
    X = df.drop(columns=["price"])
    y = df["price"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model



def predict_price_range(model: RandomForestRegressor, input_df: pd.DataFrame, mobile: UsedMobile, ai_flags: dict):
    """Predict min/max price using hybrid AI + user fallback logic."""
    
    df = input_df.copy()
    df.drop(columns=["model", "brand"], inplace=True, errors="ignore")
    
    base_price = model.predict(df)[0]

    # Merge AI + user flags
    flags = merge_ai_user_flags(ai_flags, mobile)

    # Only apply penalties when AI did NOT detect the issue
    if mobile.screen_crack and not ai_flags.get("screen_crack", False):
        base_price *= 0.7

    if mobile.panel_dot and not ai_flags.get("panel_dot", False):
        base_price *= 0.75

    if mobile.panel_line and not ai_flags.get("panel_line", False):
        base_price *= 0.7

    # Shade (AI cannot detect)
    if flags["panel_shade"]:
        base_price *= 0.75

    if flags["is_panel_changed"]:
        base_price *= 0.8

    if flags["camera_lens_ok"] is False:
        base_price *= 0.9

    if flags["fingerprint_ok"] is False:
        base_price *= 0.85

    if flags["pta_approved"] is False:
        base_price *= 0.8

    # Create +/- range
    min_price = round((base_price * 0.92) / 500) * 500
    max_price = round((base_price * 1.08) / 500) * 500

    return {
        "min_price": int(min_price),
        "max_price": int(max_price)
    }



def run_pipeline(input_mobile: UsedMobile, ai_flags: dict, db: Collection = collection):
    """
    Final integrated pipeline:
    - Fetch dataset from Mongo
    - Train model
    - Apply condition_score + hybrid AI fallback logic
    - Return price range
    """

    training_data = fetch_training_data(input_mobile.model, db)
    if not training_data:
        raise RuntimeError("No training data found for this model.")

    input_df = preprocess_input_mobile(input_mobile)
    training_df = preprocess_training_data(training_data)
    model = train_model(training_df)

    return predict_price_range(model, input_df, input_mobile, ai_flags)



# Example AI flags from damage detection (pretend values)
ai_flags = {
    "screen_crack": False,
    "panel_dot": False,
    "panel_line": False
}

# Create a sample mobile for testing
test_mobile = UsedMobile(
    brand="Google",
    model="Pixel 7A",
    ram="6GB",
    storage="128GB",

    # For AI scoring test, set condition_score manually
    condition_score=18.5,

    # User fallback flags
    is_panel_changed=False,
    screen_crack=False,
    panel_dot=False,
    panel_line=False,
    panel_shade=False,
    camera_lens_ok=True,
    fingerprint_ok=True,
    pta_approved=True,

    price=None
)

# Run price prediction
result = run_pipeline(test_mobile, ai_flags)

print("\n=== PRICE PREDICTION RESULT ===")
print(result)
