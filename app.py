import streamlit as st
import requests
import json

FASTAPI_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="IntelliFone AI", layout="wide")
st.title("📱 IntelliFone AI Dashboard")

# ============================================================
# Sidebar Navigation
# ============================================================
menu = st.sidebar.radio(
    "Select Module",
    [
        "Damage Detection (Report / JSON)",
        "Condition Scoring",
        "Price Prediction",
        "Full Verification",
        "Phone Recommendation"
    ]
)

# ============================================================
# 1. DAMAGE DETECTION (DIRECT IMAGE UPLOAD)
# ============================================================
if menu == "Damage Detection (Report / JSON)":

    st.header("📸 Damage Detection")

    mode = st.radio(
        "Select Output Type",
        ["PDF Damage Report", "Raw Damage JSON"]
    )

    st.subheader("Upload Phone Images (Any 1–6)")

    col1, col2, col3 = st.columns(3)

    with col1:
        front = st.file_uploader("Front", type=["jpg", "png"])
        back = st.file_uploader("Back", type=["jpg", "png"])

    with col2:
        left = st.file_uploader("Left", type=["jpg", "png"])
        right = st.file_uploader("Right", type=["jpg", "png"])

    with col3:
        top = st.file_uploader("Top", type=["jpg", "png"])
        bottom = st.file_uploader("Bottom", type=["jpg", "png"])

    if st.button("Analyze Damage"):

        files = {}

        for name, file in {
            "front": front,
            "back": back,
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom
        }.items():
            if file:
                files[name] = file

        if not files:
            st.error("Please upload at least one image.")
        else:
            response = requests.post(
                f"{FASTAPI_BASE_URL}/damage-detection/",
                files=files
            )

            if response.status_code == 200:

                if mode == "PDF Damage Report":
                    st.success("Damage report generated")

                    st.download_button(
                        "Download Damage Report",
                        data=response.content,
                        file_name="damage_report.pdf",
                        mime="application/pdf"
                    )

                    st.info("Raw Damage Data")
                    st.json(response.headers.get("X-Damage-Results"))

                else:
                    st.success("Damage analysis completed")
                    st.json(response.headers.get("X-Damage-Results"))

            else:
                st.error(response.text)

# ============================================================
# 2. CONDITION SCORING
# ============================================================
elif menu == "Condition Scoring":

    st.header("📊 Condition Scoring")

    st.subheader("Paste Damage JSON")
    damage_json = st.text_area("Damage JSON", height=250)

    if st.button("Compute Condition Score"):
        try:
            parsed_json = json.loads(damage_json)

            response = requests.post(
                f"{FASTAPI_BASE_URL}/condition-scoring/",
                json=parsed_json
            )

            if response.status_code == 200:
                st.success("Condition score calculated")
                st.json(response.json())
            else:
                st.error(response.text)

        except json.JSONDecodeError:
            st.error("Invalid JSON format")

# ============================================================
# 3. PRICE PREDICTION
# ============================================================
elif menu == "Price Prediction":

    st.header("💰 Price Prediction")

    col1, col2 = st.columns(2)

    with col1:
        brand = st.text_input("Brand")
        model = st.text_input("Model")
        ram = st.text_input("RAM")
        storage = st.text_input("Storage")
        condition_score = st.number_input("Condition Score", 0.0, 20.0, 15.0)

    with col2:
        pta_approved = st.checkbox("PTA Approved", True)
        camera_lens_ok = st.checkbox("Camera OK", True)
        fingerprint_ok = st.checkbox("Fingerprint OK", True)
        screen_crack = st.checkbox("Screen Crack")

    if st.button("Predict Price"):
        data = {
            "brand": brand,
            "model": model,
            "ram": ram,
            "storage": storage,
            "condition_score": condition_score,
            "pta_approved": pta_approved,
            "camera_lens_ok": camera_lens_ok,
            "fingerprint_ok": fingerprint_ok,
            "screen_crack": screen_crack
        }

        response = requests.post(
            f"{FASTAPI_BASE_URL}/price-prediction/",
            data=data
        )

        if response.status_code == 200:
            st.success("Price predicted")
            st.json(response.json())
        else:
            st.error(response.text)

# ============================================================
# 4. FULL VERIFICATION
# ============================================================
elif menu == "Full Verification":

    st.header("🔁 Full Phone Verification")

    brand = st.text_input("Brand")
    model = st.text_input("Model")
    ram = st.text_input("RAM")
    storage = st.text_input("Storage")

    st.subheader("Upload Phone Images")
    front = st.file_uploader("Front", type=["jpg", "png"])
    back = st.file_uploader("Back", type=["jpg", "png"])
    left = st.file_uploader("Left", type=["jpg", "png"])
    right = st.file_uploader("Right", type=["jpg", "png"])
    top = st.file_uploader("Top", type=["jpg", "png"])
    bottom = st.file_uploader("Bottom", type=["jpg", "png"])

    if st.button("Run Full Verification"):
        files = {}
        for name, file in {
            "front": front,
            "back": back,
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom
        }.items():
            if file:
                files[name] = file

        data = {
            "brand": brand,
            "model": model,
            "ram": ram,
            "storage": storage
        }

        response = requests.post(
            f"{FASTAPI_BASE_URL}/full-verification/",
            data=data,
            files=files
        )

        if response.status_code == 200:
            st.success("Verification completed")
            st.json(response.json())
        else:
            st.error(response.text)

# ============================================================
# 5. RECOMMENDATION ENGINE
# ============================================================
elif menu == "Phone Recommendation":

    st.header("🤖 Phone Recommendations")

    max_price = st.number_input("Max Price (PKR)", 10000, 500000, 50000)
    priority = st.selectbox(
        "Priority",
        ["camera", "gaming", "battery", "performance", "display"]
    )

    if st.button("Get Recommendations"):
        response = requests.get(
            f"{FASTAPI_BASE_URL}/recommend/",
            params={
                "max_price": max_price,
                "priority": priority
            }
        )

        if response.status_code == 200:
            st.success("Recommendations found")
            st.json(response.json())
        else:
            st.error(response.text)
