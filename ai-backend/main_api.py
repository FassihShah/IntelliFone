from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from urllib.parse import urlparse
import os
import requests
import tempfile
import uuid

from ConditionScoring.condition_scoring import compute_condition_score
from DamageDetection.Damage_Detection import analyze_phone_images
from PricePrediction.predict_price_service import ensure_price_prediction_indexes, run_pipeline
from models import UsedMobile
from report_generator import generate_damage_report, upload_report_to_supabase


app = FastAPI(title="IntelliFone AI API")
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


@app.get("/")
def read_root():
    return {"message": "Welcome to IntelliFone AI API!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


class DamageDetectionRequest(BaseModel):
    image_urls: List[str]


def validate_image_url(url: str, index: int):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(
            status_code=400,
            detail=f"Image URL at index {index} must start with http:// or https://",
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
                    detail=f"URL at index {index} did not return an image",
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
                            detail=f"Image at index {index} exceeds maximum size",
                        )
                    f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download image at index {index}: {str(e)}",
        )


def save_upload_file(upload: UploadFile, file_path: str):
    if upload.content_type and not upload.content_type.lower().startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file {upload.filename or ''} must be an image",
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
                    detail=f"Uploaded image {upload.filename or ''} exceeds maximum size",
                )
            f.write(chunk)


@app.post("/damage-detection/")
async def damage_detection(payload: DamageDetectionRequest):
    if len(payload.image_urls) == 0:
        raise HTTPException(status_code=400, detail="At least one image URL is required")

    if len(payload.image_urls) > MAX_DAMAGE_IMAGES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_DAMAGE_IMAGES} image URLs allowed")

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
                output_dir=outputs_dir,
            )

            report_path = os.path.abspath(
                os.path.join(reports_dir, f"damage_report_{uuid.uuid4()}.pdf")
            )

            generate_damage_report(
                damages=result["damages"],
                output_dir=outputs_dir,
                report_path=report_path,
            )

            report_url = upload_report_to_supabase(report_path)
            scoring = compute_condition_score(result)

            return {
                "pdf_url": report_url,
                "condition_score": scoring["condition_score"],
                "ai_detected": scoring["ai_detected"],
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Damage detection failed: {str(e)}")


@app.post("/condition-scoring/")
async def condition_scoring(damage_json: dict):
    return compute_condition_score(damage_json)


@app.post("/price-prediction/")
async def price_prediction(
    brand: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    ram: Optional[str] = Form(None),
    storage: Optional[str] = Form(None),
    condition_score: float = Form(...),
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
    ai_panel_line: bool = Form(False),
):
    ai_flags = {
        "screen_crack": ai_screen_crack,
        "panel_dot": ai_panel_dot,
        "panel_line": ai_panel_line,
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
        pta_approved=pta_approved,
    )

    return run_pipeline(mobile, ai_flags)


@app.post("/full-verification/")
async def full_verification(
    brand: str = Form(...),
    model: str = Form(...),
    ram: str = Form(...),
    storage: str = Form(...),
    is_panel_changed: bool = Form(False),
    screen_crack: bool = Form(False),
    panel_dot: bool = Form(False),
    panel_line: bool = Form(False),
    panel_shade: bool = Form(False),
    camera_lens_ok: bool = Form(True),
    fingerprint_ok: bool = Form(True),
    pta_approved: bool = Form(True),
    front: Optional[UploadFile] = File(None),
    back: Optional[UploadFile] = File(None),
    left: Optional[UploadFile] = File(None),
    right: Optional[UploadFile] = File(None),
    top: Optional[UploadFile] = File(None),
    bottom: Optional[UploadFile] = File(None),
):
    uploads = {}

    with tempfile.TemporaryDirectory(prefix="intellifone_full_") as request_dir:
        uploads_dir = os.path.join(request_dir, "uploads")
        outputs_dir = os.path.join(request_dir, "outputs")
        os.makedirs(uploads_dir, exist_ok=True)

        for side, img in {
            "front": front,
            "back": back,
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
        }.items():
            if img:
                file_id = f"{uuid.uuid4()}.jpg"
                file_path = os.path.join(uploads_dir, file_id)
                save_upload_file(img, file_path)
                uploads[side] = file_path
            else:
                uploads[side] = None

        damage_result = analyze_phone_images(
            YOLO_MODEL_PATH,
            uploads,
            show_output=False,
            save_output=False,
            output_dir=outputs_dir,
        )

        scoring = compute_condition_score(damage_result)
        ai_flags = scoring["ai_detected"]
        condition_score = scoring["condition_score"]

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
            images=[],
        )

        price_range = run_pipeline(mobile, ai_flags)

        return {
            "damage_detection": damage_result,
            "condition_score": condition_score,
            "ai_flags": ai_flags,
            "price_range": price_range,
            "mobile_info": mobile.model_dump(),
        }
