from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import requests
import os
import tempfile
import uuid
from pydantic import BaseModel
from urllib.parse import urlparse
from report_generator import generate_damage_report, upload_report_to_supabase

# --- Import your modules ---
from models import UsedMobile
from DamageDetection.Damage_Detection import analyze_phone_images
from PricePrediction.predict_price_service import ensure_price_prediction_indexes, run_pipeline
from ConditionScoring.condition_scoring import compute_condition_score 
from RecommendationEngine.recommendation_service import ensure_recommendation_indexes, get_recommendations
from models import ChatRequest, ChatResponse, ChatHistoryResponse
from ChatBot.chatbot import generate_reply
from ChatBot.crud import (
    create_conversation,
    get_chat_history,
    get_chat_history_formatted,
    save_message
)

app = FastAPI(title="IntelliFone AI Backend")
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best3.pt")
MAX_DAMAGE_IMAGES = 6
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))


def get_allowed_origins():
    raw_origins = os.getenv("ALLOWED_ORIGINS", "")
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return origins or ["http://localhost:3000", "http://127.0.0.1:3000"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def validate_startup_configuration():
    required_env_vars = [
        "MONGO_CONNECTION_STRING",
        "DEEPSEEK_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
    ]
    missing = [name for name in required_env_vars if not os.getenv(name)]

    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {missing_list}")

    if not os.path.exists(YOLO_MODEL_PATH):
        raise RuntimeError(f"YOLO model file not found: {YOLO_MODEL_PATH}")


@app.on_event("startup")
def startup_checks():
    validate_startup_configuration()
    ensure_price_prediction_indexes()
    ensure_recommendation_indexes()



@app.get("/")
def read_root():
    return {"message": "Welcome to IntelliFone!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


# # ============================================================
# #  ENDPOINT 1 — DAMAGE DETECTION
# # ============================================================
class DamageDetectionRequest(BaseModel):
    image_urls: List[str]  # max 6 URLs


def validate_image_url(url: str, index: int):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(
            status_code=400,
            detail=f"Image URL at index {index} must start with http:// or https://"
        )


def download_image(url: str, file_path: str, index: int):
    validate_image_url(url, index)

    try:
        with requests.get(url, timeout=10, stream=True) as response:
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if content_type and not content_type.lower().startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"URL at index {index} did not return an image"
                )

            downloaded = 0
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > MAX_IMAGE_BYTES:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Image at index {index} exceeds maximum size"
                        )
                    f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download image at index {index}: {str(e)}"
        )


def save_upload_file(upload: UploadFile, file_path: str):
    if upload.content_type and not upload.content_type.lower().startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file {upload.filename or ''} must be an image"
        )

    written = 0
    with open(file_path, "wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_IMAGE_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Uploaded image {upload.filename or ''} exceeds maximum size"
                )
            f.write(chunk)


@app.post("/damage-detection/")
async def damage_detection(payload: DamageDetectionRequest):

    if len(payload.image_urls) == 0:
        raise HTTPException(status_code=400, detail="At least one image URL is required")

    if len(payload.image_urls) > MAX_DAMAGE_IMAGES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_DAMAGE_IMAGES} image URLs allowed")

    # Expected sides (order-based mapping)
    sides = ["front", "back", "left", "right", "top", "bottom"]
    saved = {side: None for side in sides}

    with tempfile.TemporaryDirectory(prefix="intellifone_damage_") as request_dir:
        uploads_dir = os.path.join(request_dir, "uploads")
        outputs_dir = os.path.join(request_dir, "outputs")
        reports_dir = os.path.join(request_dir, "reports")
        os.makedirs(uploads_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        try:
            # Download images
            for idx, url in enumerate(payload.image_urls):
                ext = os.path.splitext(urlparse(url).path)[1] or ".jpg"
                file_name = f"{uuid.uuid4()}{ext}"
                file_path = os.path.join(uploads_dir, file_name)
                download_image(url, file_path, idx)
                saved[sides[idx]] = file_path

            result = analyze_phone_images(
                YOLO_MODEL_PATH,
                saved,
                show_output=False,
                save_output=True,
                output_dir=outputs_dir
            )

            report_path = os.path.abspath(
                os.path.join(reports_dir, f"damage_report_{uuid.uuid4()}.pdf")
            )

            generate_damage_report(
                damages=result["damages"],
                output_dir=outputs_dir,
                report_path=report_path
            )

            report_url = upload_report_to_supabase(report_path)
            scoring = compute_condition_score(result)

            return {
                "pdf_url": report_url,
                "condition_score": scoring["condition_score"],
                "ai_detected": scoring["ai_detected"]
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Damage detection failed: {str(e)}")


# ============================================================
#  ENDPOINT 2 — CONDITION SCORING
# ============================================================
@app.post("/condition-scoring/")
async def condition_scoring(damage_json: dict):
    result = compute_condition_score(damage_json)
    return result



# # ============================================================
# #  ENDPOINT 3 — PRICE PREDICTION (AI + USER FALLBACK)
# # ============================================================
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

    with tempfile.TemporaryDirectory(prefix="intellifone_full_") as request_dir:
        uploads_dir = os.path.join(request_dir, "uploads")
        outputs_dir = os.path.join(request_dir, "outputs")
        os.makedirs(uploads_dir, exist_ok=True)

        for side, img in {
            "front": front, "back": back, "left": left,
            "right": right, "top": top, "bottom": bottom
        }.items():
            if img:
                file_id = f"{uuid.uuid4()}.jpg"
                file_path = os.path.join(uploads_dir, file_id)
                save_upload_file(img, file_path)
                uploads[side] = file_path
            else:
                uploads[side] = None

        # -------------------------------
        # Run YOLO Damage Detection
        # -------------------------------
        damage_result = analyze_phone_images(
            YOLO_MODEL_PATH,
            uploads,
            show_output=False,
            save_output=False,
            output_dir=outputs_dir
        )

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
            images=[]
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
            "mobile_info": mobile.model_dump()
        }



# ============================================================
#  ENDPOINT 5 — PHONE RECOMMENDATIONS
# ============================================================
@app.get("/recommend/")
async def recommend_phones(max_price: float, priority: str):
    return get_recommendations(max_price, priority)



# ============================================================
#  ENDPOINT 6 — CHATBOT INTERFACE
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    conversation_id = req.conversation_id

    if not conversation_id:
        conversation_id = create_conversation(
            req.user_id, req.message
        )

    history = get_chat_history(conversation_id)

    reply = generate_reply(history, req.message)

    save_message(conversation_id, req.user_id, "user", req.message)
    save_message(conversation_id, req.user_id, "assistant", reply)

    return {
        "conversation_id": conversation_id,
        "reply": reply
    }
# ============================================================
#  ENDPOINT 7 — get all messages in a conversation
@app.get("/chat/{conversation_id}", response_model=ChatHistoryResponse)
async def get_chat(conversation_id: str):
    history = get_chat_history_formatted(conversation_id)
    return history
