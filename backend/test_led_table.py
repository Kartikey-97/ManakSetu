import time
from pathlib import Path
import pdfplumber
import pymupdf

pdf_path = Path("/Users/kartikeygupta/Desktop/Hackathons/sih_26108/backend/uploads/850189f2-af13-4ec5-8878-bf9baf6df897/Tendernotice_1.pdf")

print("=== PDFPLUMBER ===")
with pdfplumber.open(pdf_path) as pdf:
    print(f"Pages: {len(pdf.pages)}")
    print(pdf.pages[0].extract_text()[:500])

print("\n=== PYMUPDF ===")
doc = pymupdf.open(pdf_path)
print(f"Pages: {len(doc)}")
print(doc[0].get_text()[:500])
doc.close()
