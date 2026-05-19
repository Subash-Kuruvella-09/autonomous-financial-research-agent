"""
Utility for generating professional-grade PDF documents from research reports using ReportLab.
"""

import os
import re
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))

def strip_markdown(text: str) -> str:
    """
    Removes common markdown artifacts to ensure clean ReportLab rendering.
    """
    # Remove Bold/Italic
    text = re.sub(r'\*\*+(.*?)\*\*+', r'\1', text)
    text = re.sub(r'\*+(.*?)\*+', r'\1', text)
    text = re.sub(r'__+(.*?)__+', r'\1', text)
    text = re.sub(r'_+(.*?)_+', r'\1', text)
    
    # Remove Headings symbols
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Remove blockquotes
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    
    # Remove horizontal rules
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\*{3,}$', '', text, flags=re.MULTILINE)
    
    return text.strip()

def generate_pdf(report_text: str, company_name: str) -> str:
    """
    Generates a professional financial report PDF using ReportLab Platypus.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clean the company name for filename
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', company_name)
    filename = f"{safe_name}_Report_{datetime.now().strftime('%H%M%S')}.pdf"
    filepath = os.path.join(OUTPUT_DIR, filename)

    doc = SimpleDocTemplate(
        filepath, 
        pagesize=letter,
        rightMargin=72, leftMargin=72,
        topMargin=72, bottomMargin=18
    )

    styles = getSampleStyleSheet()
    
    # Custom Title Style
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor("#1a5f7a"),
        spaceAfter=30,
        alignment=1 # Center
    )

    # Custom Heading Style
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor("#1a5f7a"),
        spaceBefore=20,
        spaceAfter=12,
        borderPadding=(0, 0, 2, 0),
        borderWidth=0,
        borderColor=colors.HexColor("#bdc3c7")
    )

    # Custom Body Style
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        alignment=4 # Justify
    )

    story = []

    # Title
    story.append(Paragraph(f"{company_name} Research Report", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y')}", styles["Normal"]))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Spacer(1, 12))

    # Parse report text into sections
    # Split by double newlines or single newlines with headings
    blocks = re.split(r'\n\s*\n', report_text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Check if this block is a heading (Markdown style or just short)
        is_heading = block.startswith('#') or (len(block) < 80 and '\n' not in block)
        
        # Strip markdown artifacts
        clean_content = strip_markdown(block)
        
        if is_heading:
            story.append(Paragraph(clean_content, heading_style))
        else:
            story.append(Paragraph(clean_content, body_style))
            story.append(Spacer(1, 12))

    # Footer placeholder (SimpleDocTemplate doesn't do footers easily without Canvas)
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("---", styles["Normal"]))
    story.append(Paragraph("Financial Research | Verified Analysis", styles["Italic"]))

    doc.build(story)

    return filepath
