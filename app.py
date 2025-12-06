from fastapi import FastAPI, UploadFile, File, Form
from typing import List, Optional
import os
import shutil
import uuid

# --- Import your modules ---
from models import UsedMobile
from DamageDetection.Damage_Detection import analyze_phone_images
from ConditionScoring.condition_scoring import compute_condition_score
from PricePrediction.predict_price_service import run_pipeline, merge_ai_user_flags
from RecommendationEngine.recommendation_service import get_recommendations
             

app = FastAPI(title="IntelliFone AI Backend")


# ============================================================
#  ENDPOINT 1 — DAMAGE DETECTION
# ============================================================
@app.post("/damage-detection/")
async def damage_detection(
    front: Optional[UploadFile] = File(None),
    back: Optional[UploadFile] = File(None),
    left: Optional[UploadFile] = File(None),
    right: Optional[UploadFile] = File(None),
    top: Optional[UploadFile] = File(None),
    bottom: Optional[UploadFile] = File(None),
):
    images = {
        "front": front,
        "back": back,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom
    }

    saved = {}

    os.makedirs("uploads", exist_ok=True)

    # Save all uploaded images
    for side, file in images.items():
        if file:
            file_id = f"{uuid.uuid4()}.jpg"
            file_path = os.path.join("uploads", file_id)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved[side] = file_path
        else:
            saved[side] = None

    # Run YOLO model
    model_path = os.path.join(os.path.dirname(__file__), "best2.pt")
    result = analyze_phone_images(model_path, saved, show_output=False, save_output=False)

    return result



# ============================================================
#  ENDPOINT 2 — CONDITION SCORING
# ============================================================
@app.post("/condition-scoring/")
async def condition_scoring(damage_json: dict):
    result = compute_condition_score(damage_json)
    return result



# ============================================================
#  ENDPOINT 3 — PRICE PREDICTION (AI + USER FALLBACK)
# ============================================================
@app.post("/price-prediction/")
async def price_prediction(
    brand: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    ram: Optional[str] = Form(None),
    storage: Optional[str] = Form(None),
    condition_score: float = Form(...),

    # User fallback fields
    is_panel_changed: bool = Form(False),
    screen_crack: bool = Form(False),
    panel_dot: bool = Form(False),
    panel_line: bool = Form(False),
    panel_shade: bool = Form(False),
    camera_lens_ok: bool = Form(True),
    fingerprint_ok: bool = Form(True),
    pta_approved: bool = Form(True),

    ai_screen_crack: bool = Form(False),
    ai_panel_dot: bool = Form(False),
    ai_panel_line: bool = Form(False)
):
    ai_flags = {
        "screen_crack": ai_screen_crack,
        "panel_dot": ai_panel_dot,
        "panel_line": ai_panel_line
    }

    mobile = UsedMobile(
        brand=brand,
        model=model,
        ram=ram,
        storage=storage,
        condition_score=condition_score,

        is_panel_changed=is_panel_changed,
        screen_crack=screen_crack,
        panel_dot=panel_dot,
        panel_line=panel_line,
        panel_shade=panel_shade,
        camera_lens_ok=camera_lens_ok,
        fingerprint_ok=fingerprint_ok,
        pta_approved=pta_approved
    )

    price_range = run_pipeline(mobile, ai_flags)

    return price_range



# ============================================================
#  ENDPOINT 4 — FULL VERIFICATION PIPELINE
# ============================================================
@app.post("/full-verification/")
async def full_verification(
    brand: str = Form(...),
    model: str = Form(...),
    ram: str = Form(...),
    storage: str = Form(...),

    # User fallback inputs
    is_panel_changed: bool = Form(False),
    screen_crack: bool = Form(False),
    panel_dot: bool = Form(False),
    panel_line: bool = Form(False),
    panel_shade: bool = Form(False),
    camera_lens_ok: bool = Form(True),
    fingerprint_ok: bool = Form(True),
    pta_approved: bool = Form(True),

    # Images
    front: Optional[UploadFile] = File(None),
    back: Optional[UploadFile] = File(None),
    left: Optional[UploadFile] = File(None),
    right: Optional[UploadFile] = File(None),
    top: Optional[UploadFile] = File(None),
    bottom: Optional[UploadFile] = File(None),
):
    # -------------------------------
    # Save images
    # -------------------------------
    uploads = {}
    os.makedirs("uploads", exist_ok=True)

    for side, img in {
        "front": front, "back": back, "left": left,
        "right": right, "top": top, "bottom": bottom
    }.items():
        if img:
            file_id = f"{uuid.uuid4()}.jpg"
            file_path = f"uploads/{file_id}"
            with open(file_path, "wb") as f:
                shutil.copyfileobj(img.file, f)
            uploads[side] = file_path
        else:
            uploads[side] = None

    # -------------------------------
    # Run YOLO Damage Detection
    # -------------------------------
    model_path = os.path.join(os.path.dirname(__file__), "best2.pt")
    damage_result = analyze_phone_images(model_path, uploads, show_output=False, save_output=False)

    # -------------------------------
    # Condition Scoring
    # -------------------------------
    scoring = compute_condition_score(damage_result)
    ai_flags = scoring["ai_detected"]
    condition_score = scoring["condition_score"]

    # -------------------------------
    # Build UsedMobile object
    # -------------------------------
    mobile = UsedMobile(
        brand=brand,
        model=model,
        ram=ram,
        storage=storage,
        condition_score=condition_score,
        is_panel_changed=is_panel_changed,
        screen_crack=screen_crack,
        panel_dot=panel_dot,
        panel_line=panel_line,
        panel_shade=panel_shade,
        camera_lens_ok=camera_lens_ok,
        fingerprint_ok=fingerprint_ok,
        pta_approved=pta_approved,
        images=list(uploads.values())
    )

    # -------------------------------
    # Price Prediction
    # -------------------------------
    price_range = run_pipeline(mobile, ai_flags)

    # -------------------------------
    # Final Output
    # -------------------------------
    return {
        "damage_detection": damage_result,
        "condition_score": condition_score,
        "ai_flags": ai_flags,
        "price_range": price_range,
        "mobile_info": mobile.model_dump(),
        "uploaded_images": uploads
    }



# ============================================================
#  ENDPOINT 5 — PHONE RECOMMENDATIONS
# ============================================================
@app.get("/recommend/")
async def recommend_phones(max_price: float, priority: str):
    return get_recommendations(max_price, priority)
