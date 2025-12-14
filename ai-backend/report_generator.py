from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

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
