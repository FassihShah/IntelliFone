import os
import uuid

import requests
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer

load_dotenv()

def generate_damage_report(damages, output_dir, report_path):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(report_path, pagesize=A4)
    story = []

    story.append(Paragraph("<b>Mobile Damage Detection Report</b>", styles["Title"]))
    story.append(Spacer(1, 20))

    for side, damage in damages.items():
        story.append(Paragraph(f"<b>{side.capitalize()} Side</b>", styles["Heading2"]))
        story.append(Spacer(1, 10))

        output_img = os.path.join(output_dir, f"{side}_output.jpg")
        if os.path.exists(output_img):
            story.append(Image(output_img, width=250, height=250))
            story.append(Spacer(1, 10))

        for dtype, values in damage.items():
            for v in values:
                metric = ", ".join(f"{k}: {val}" for k, val in v.items())
                story.append(Paragraph(f"{dtype.capitalize()} → {metric}", styles["Normal"]))

        story.append(Spacer(1, 25))

    doc.build(story)


def upload_report_to_supabase(report_path):
    supabase_url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    bucket_name = os.getenv("SUPABASE_REPORTS_BUCKET", "phone-reports")
    folder_name = os.getenv("SUPABASE_REPORTS_FOLDER", "damage_reports")

    if not supabase_url or not service_role_key:
        raise RuntimeError(
            "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY for report upload"
        )

    file_name = f"{folder_name.rstrip('/')}/report_{uuid.uuid4()}.pdf"
    upload_url = f"{supabase_url.rstrip('/')}/storage/v1/object/{bucket_name}/{file_name}"
    headers = {
        "Authorization": f"Bearer {service_role_key}",
        "apikey": service_role_key,
        "Content-Type": "application/pdf",
        "x-upsert": "true",
    }

    with open(report_path, "rb") as pdf_file:
        response = requests.post(upload_url, headers=headers, data=pdf_file, timeout=60)

    if response.status_code >= 300:
        raise RuntimeError(
            f"Supabase upload failed ({response.status_code}): {response.text}"
        )

    return f"{supabase_url.rstrip('/')}/storage/v1/object/public/{bucket_name}/{file_name}"
