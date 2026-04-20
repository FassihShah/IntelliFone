"""
IntelliFone — AI Damage Detection Report Generator
Branded, premium PDF report matching IntelliFone's dark tech aesthetic.
"""

import os
import uuid
import requests
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image, Spacer, Table,
    TableStyle, HRFlowable, KeepTogether
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.lib.colors import HexColor, Color
from datetime import datetime

# ─── Brand Palette ───────────────────────────────────────────────────────────
DARK_BG        = HexColor("#0A0E1A")   # page background
SURFACE        = HexColor("#111827")   # card background
SURFACE_2      = HexColor("#1C2333")   # secondary card
ACCENT         = HexColor("#3B82F6")   # electric blue (IntelliFone primary)
ACCENT_GLOW    = HexColor("#60A5FA")   # lighter blue
SUCCESS        = HexColor("#10B981")   # emerald green — no damage
WARNING        = HexColor("#F59E0B")   # amber — minor damage
DANGER         = HexColor("#EF4444")   # red — major damage
YELLOW         = HexColor("#FACC15")   # yellow — highlight accent
TEXT_PRIMARY   = HexColor("#F9FAFB")   # near-white
TEXT_SECONDARY = HexColor("#9CA3AF")   # muted gray
TEXT_ACCENT    = HexColor("#BFDBFE")   # light blue
BORDER         = HexColor("#1F2937")   # subtle border
DIVIDER        = HexColor("#374151")   # stronger divider

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm


# ─── Custom Flowables ─────────────────────────────────────────────────────────

class ColorRect(Flowable):
    """A filled rectangle, used for section headers and dividers."""
    def __init__(self, width, height, color, radius=4):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.color = color
        self.radius = radius

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.roundRect(0, 0, self.width, self.height, self.radius, fill=1, stroke=0)


class GradientHeader(Flowable):
    """Full-width dark header band with logo area and report title."""
    def __init__(self, width, title, subtitle, report_id):
        Flowable.__init__(self)
        self.width = width
        self.height = 52 * mm
        self.title = title
        self.subtitle = subtitle
        self.report_id = report_id

    def draw(self):
        c = self.canv
        w, h = self.width, self.height

        # Background panel
        c.setFillColor(SURFACE)
        c.roundRect(0, 0, w, h, 8, fill=1, stroke=0)

        # Left accent bar
        c.setFillColor(ACCENT)
        c.rect(0, 0, 3, h, fill=1, stroke=0)

        # Top right corner ornament
        c.setFillColor(ACCENT)
        c.setFillAlpha(0.08)
        c.circle(w - 10 * mm, h, 28 * mm, fill=1, stroke=0)
        c.setFillAlpha(1.0)

        # "IF" Logo badge
        badge_size = 14 * mm
        badge_x = 6 * mm
        badge_y = h - badge_size - 8 * mm
        c.setFillColor(ACCENT)
        c.roundRect(badge_x, badge_y, badge_size, badge_size, 4, fill=1, stroke=0)
        c.setFillColor(TEXT_PRIMARY)
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(badge_x + badge_size / 2, badge_y + badge_size / 2 - 3.5, "IF")

        # IntelliFone wordmark
        c.setFillColor(TEXT_PRIMARY)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(badge_x + badge_size + 4 * mm, badge_y + badge_size - 5 * mm, "IntelliFone")
        c.setFillColor(TEXT_ACCENT)
        c.setFont("Helvetica", 7.5)
        c.drawString(badge_x + badge_size + 4 * mm, badge_y + 2 * mm, "AI-Powered Phone Verification")

        # Report title
        c.setFillColor(TEXT_PRIMARY)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(6 * mm, h / 2 - 8 * mm, self.title)

        # Subtitle
        c.setFillColor(YELLOW)
        c.setFont("Helvetica", 9)
        c.drawString(6 * mm, h / 2 - 14 * mm, self.subtitle)

        # Report ID pill (bottom right)
        pill_text = f"Report ID: {self.report_id}"
        c.setFont("Helvetica", 7.5)
        pill_w = c.stringWidth(pill_text, "Helvetica", 7.5) + 12
        pill_x = w - pill_w - 6 * mm
        pill_y = 4 * mm
        c.setFillColor(DARK_BG)
        c.roundRect(pill_x, pill_y, pill_w, 6 * mm, 3, fill=1, stroke=0)
        c.setFillColor(TEXT_SECONDARY)
        c.drawString(pill_x + 6, pill_y + 3.5, pill_text)

        # Generated date (bottom left)
        now = datetime.now().strftime("%B %d, %Y · %H:%M")
        c.setFillColor(TEXT_SECONDARY)
        c.setFont("Helvetica", 7.5)
        c.drawString(6 * mm, 5.5 * mm, f"Generated: {now}")


class SectionHeader(Flowable):
    """A styled section header pill."""
    def __init__(self, width, label, icon_char="◆"):
        Flowable.__init__(self)
        self.width = width
        self.height = 9 * mm
        self.label = label
        self.icon_char = icon_char

    def draw(self):
        c = self.canv
        w, h = self.width, self.height

        # Background strip
        c.setFillColor(SURFACE_2)
        c.roundRect(0, 0, w, h, 4, fill=1, stroke=0)

        # Left accent line
        c.setFillColor(ACCENT)
        c.rect(0, 0, 2.5, h, fill=1, stroke=0)

        # Icon dot
        c.setFillColor(YELLOW)
        c.circle(8 * mm, h / 2, 2, fill=1, stroke=0)

        # Label
        c.setFillColor(YELLOW)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(11 * mm, h / 2 - 3, self.label.upper())


class DamageBadge(Flowable):
    """A severity badge — NONE / MINOR / MODERATE / MAJOR."""
    COLORS = {
        "none":     (SUCCESS,  "#D1FAE5", "NO DAMAGE"),
        "minor":    (WARNING,  "#FEF3C7", "MINOR"),
        "moderate": (WARNING,  "#FEF3C7", "MODERATE"),
        "major":    (DANGER,   "#FEE2E2", "MAJOR"),
        "unknown":  (TEXT_SECONDARY, "#374151", "UNKNOWN"),
    }

    def __init__(self, severity="unknown"):
        Flowable.__init__(self)
        self.severity = severity.lower()
        self.width = 28 * mm
        self.height = 6 * mm

    def draw(self):
        c = self.canv
        accent_color, _, label = self.COLORS.get(self.severity, self.COLORS["unknown"])
        c.setFillColor(accent_color)
        c.setFillAlpha(0.15)
        c.roundRect(0, 0, self.width, self.height, 3, fill=1, stroke=0)
        c.setFillAlpha(1.0)
        c.setStrokeColor(accent_color)
        c.setLineWidth(0.5)
        c.roundRect(0, 0, self.width, self.height, 3, fill=0, stroke=1)
        c.setFillColor(accent_color)
        c.setFont("Helvetica-Bold", 7)
        c.drawCentredString(self.width / 2, self.height / 2 - 2.5, label)


class FooterCanvas(pdfcanvas.Canvas):
    """Canvas override to add footer to every page."""
    def __init__(self, filename, **kwargs):
        pdfcanvas.Canvas.__init__(self, filename, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_footer(num_pages)
            pdfcanvas.Canvas.showPage(self)
        pdfcanvas.Canvas.save(self)

    def _draw_footer(self, page_count):
        self.saveState()
        y = 10 * mm

        # Footer line
        self.setStrokeColor(DIVIDER)
        self.setLineWidth(0.5)
        self.line(MARGIN, y + 4 * mm, PAGE_W - MARGIN, y + 4 * mm)

        # Left: brand
        self.setFillColor(TEXT_SECONDARY)
        self.setFont("Helvetica", 7)
        self.drawString(MARGIN, y, "IntelliFone · AI-Powered Mobile Verification")

        # Center: tagline
        self.setFillColor(ACCENT)
        self.setFont("Helvetica-Bold", 7)
        self.drawCentredString(PAGE_W / 2, y, "intellifone.vercel.app")

        # Right: page number
        self.setFillColor(TEXT_SECONDARY)
        self.setFont("Helvetica", 7)
        page_num = self._pageNumber
        self.drawRightString(PAGE_W - MARGIN, y, f"Page {page_num} of {page_count}")

        self.restoreState()


# ─── Report Builder ───────────────────────────────────────────────────────────

def _damage_severity(damage: dict) -> str:
    """Guess severity from damage data."""
    for dtype, values in damage.items():
        if dtype == "crack" and values:
            return "major"
        if dtype == "crack" and values:
            return "minor"
        if dtype == "line" and values:
            for v in values:
                length = v.get("length_px", 0)
                if length > 500:
                    return "moderate"
                return "minor"
    return "none"


def _status_icon(severity: str) -> str:
    return {"none": "✓", "minor": "!", "moderate": "!!", "major": "✗"}.get(severity, "?")


def generate_damage_report(damages: dict, output_dir: str, report_path: str):
    """Generate a beautifully branded IntelliFone damage detection report."""
    report_id = str(uuid.uuid4())[:8].upper()

    doc = SimpleDocTemplate(
        report_path,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=20 * mm,
    )

    # ── Styles ────────────────────────────────────────────────────────────────
    metric_label = ParagraphStyle(
        "metric_label",
        fontName="Helvetica-Bold",
        fontSize=8,
        textColor=TEXT_ACCENT,
        leading=12,
    )
    metric_value = ParagraphStyle(
        "metric_value",
        fontName="Helvetica",
        fontSize=8,
        textColor=TEXT_PRIMARY,
        leading=12,
    )
    side_title = ParagraphStyle(
        "side_title",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=TEXT_PRIMARY,
        leading=14,
    )
    no_damage = ParagraphStyle(
        "no_damage",
        fontName="Helvetica",
        fontSize=8.5,
        textColor=SUCCESS,
        leading=12,
    )

    content_w = PAGE_W - 2 * MARGIN
    story = []

    # ── Cover Header ──────────────────────────────────────────────────────────
    story.append(GradientHeader(
        content_w,
        title="Mobile Damage Detection Report",
        subtitle="Comprehensive AI-powered physical inspection analysis",
        report_id=report_id,
    ))
    story.append(Spacer(1, 5 * mm))

    # ── Summary ───────────────────────────────────────────────────────────────
    story.append(SectionHeader(content_w, "Inspection Summary", "◆"))
    story.append(Spacer(1, 3 * mm))

    total_issues = sum(
        sum(len(v) for v in d.values()) for d in damages.values()
    )
    sides_total = len(damages)
    col_w = content_w / 2

    summary_data = [
        [
            Paragraph("<b>Issues Detected</b>", metric_label),
            Paragraph("<b>Sides Inspected</b>", metric_label),
        ],
        [
            Paragraph(
                f'<font size="20"><b>{total_issues}</b></font>',
                ParagraphStyle("sc2", fontName="Helvetica-Bold", fontSize=20,
                               textColor=(DANGER if total_issues > 3 else (WARNING if total_issues > 0 else SUCCESS)),
                               leading=24)
            ),
            Paragraph(
                f'<font size="20"><b>{sides_total}</b></font>',
                ParagraphStyle("sc3", fontName="Helvetica-Bold", fontSize=20, textColor=ACCENT, leading=24)
            ),
        ]
    ]

    summary_table = Table(summary_data, colWidths=[col_w] * 2)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), SURFACE),
        ("BACKGROUND",   (0, 0), (-1, 0),  SURFACE_2),
        ("BOX",          (0, 0), (-1, -1), 0.5, DIVIDER),
        ("LINEAFTER",    (0, 0), (0, -1),  0.5, DIVIDER),
        ("LINEBEFORE",   (0, 0), (0, -1),  1.5, ACCENT),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 5 * mm))

    # ── Per-Side Inspection ───────────────────────────────────────────────────
    story.append(SectionHeader(content_w, "Side-by-Side Inspection", "◆"))
    story.append(Spacer(1, 3 * mm))

    for side, damage in damages.items():
        severity = _damage_severity(damage)
        severity_color = {"none": SUCCESS, "minor": WARNING, "moderate": WARNING, "major": DANGER}.get(severity, TEXT_SECONDARY)
        icon = _status_icon(severity)

        img_path = os.path.join(output_dir, f"{side}_output.jpg")
        has_image = os.path.exists(img_path)

        # Image + info layout
        img_col = 55 * mm
        info_col = content_w - img_col - 4 * mm

        # Build damage detail rows
        detail_rows = []
        has_issues = False
        for dtype, values in damage.items():
            for v in values:
                has_issues = True
                metric_str = ", ".join(f"{k}: {val}" for k, val in v.items())
                detail_rows.append((dtype.replace("_", " ").title(), metric_str))

        # Info panel content
        info_content = [
            [
                Paragraph(f"<b>{side.replace('_', ' ').title()} Side</b>", side_title),
                Paragraph(f'<font color="{severity_color.hexval()}"><b>{icon} {severity.upper()}</b></font>', ParagraphStyle("sev", fontName="Helvetica-Bold", fontSize=9, textColor=severity_color, leading=12, alignment=TA_RIGHT)),
            ]
        ]

        if not has_issues:
            info_content.append([
                Paragraph("✓ No damage detected on this side.", no_damage),
                Paragraph(""),
            ])
        else:
            for dtype_label, metric_str in detail_rows:
                info_content.append([
                    Paragraph(f"<b>{dtype_label}</b>", metric_label),
                    Paragraph(metric_str, metric_value),
                ])

        info_table = Table(
            info_content,
            colWidths=[info_col * 0.4, info_col * 0.6],
        )
        info_table.setStyle(TableStyle([
            ("SPAN",         (0, 0), (-1, 0)),
            ("BACKGROUND",   (0, 0), (-1, -1), SURFACE),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("LINEBELOW",    (0, 0), (-1, 0),  0.5, DIVIDER),
        ]))

        # Image cell
        if has_image:
            img = Image(img_path, width=img_col - 2 * mm, height=42 * mm)
            img_cell = [[img]]
        else:
            # Placeholder
            img_cell = [[Paragraph(
                '<font color="#374151">No image</font>',
                ParagraphStyle("ph", fontName="Helvetica", fontSize=8, textColor=DIVIDER, alignment=TA_CENTER)
            )]]

        img_table = Table(img_cell, colWidths=[img_col])
        img_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), DARK_BG),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
            ("LEFTPADDING",  (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))

        # Side card: image left | info right
        row_data = [[img_table, info_table]]
        row_table = Table(row_data, colWidths=[img_col + 2 * mm, info_col])
        row_table.setStyle(TableStyle([
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING",   (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
            # Left accent colour based on severity
            ("LINEBEFORE",   (0, 0), (0, -1),  2.5, severity_color),
            ("BOX",          (0, 0), (-1, -1),  0.5, DIVIDER),
        ]))

        story.append(KeepTogether([row_table, Spacer(1, 3 * mm)]))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    disclaimer = ParagraphStyle(
        "disclaimer",
        fontName="Helvetica",
        fontSize=7,
        textColor=TEXT_SECONDARY,
        leading=11,
    )
    story.append(HRFlowable(width=content_w, thickness=0.5, color=DIVIDER))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "This report is generated by IntelliFone's AI damage detection system and is provided for informational "
        "purposes only. Results are based on computer vision analysis and may not capture all damage. "
        "IntelliFone recommends a physical inspection for final valuation. © IntelliFone · intellifone.vercel.app",
        disclaimer,
    ))

    # ── Build PDF ─────────────────────────────────────────────────────────────
    doc.build(story, canvasmaker=FooterCanvas)
    print(f"✅  Report saved → {report_path}")


# ─── Supabase Upload (unchanged API) ──────────────────────────────────────────

def upload_report_to_supabase(report_path):
    supabase_url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    bucket_name = os.getenv("SUPABASE_REPORTS_BUCKET", "phone-reports")
    folder_name = os.getenv("SUPABASE_REPORTS_FOLDER", "damage_reports")

    if not supabase_url or not service_role_key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

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
        raise RuntimeError(f"Upload failed ({response.status_code}): {response.text}")

    return f"{supabase_url.rstrip('/')}/storage/v1/object/public/{bucket_name}/{file_name}"


# # ─── Demo ─────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     sample_damages = {
#         "front":  {"crack": [{"length_px": 120, "count": 2}]},
#         "back":   {"crack": [{"length_px": 45}]},
#         "left":   {"line":    [{"length_px": 697.5, "severity": 0.38}]},
#         "right":  {"crack": [{"length_px": 80}]},
#         "top":    {},
#         "bottom": {},
#     }
#     out = "/home/claude"
#     generate_damage_report(sample_damages, out, "/home/claude/sample_report.pdf")
