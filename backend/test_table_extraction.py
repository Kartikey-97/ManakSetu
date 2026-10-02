import time
from pathlib import Path
import pdfplumber
import pymupdf

pdf_path = Path("/Users/kartikeygupta/Desktop/Hackathons/sih_26108/backend/uploads/ef70ec28-66f1-4c99-9134-ef1d069c5877/hindi:english tender.pdf")

# Let's test page 10 (usually has some tables or dense text)
print("=== PDFPLUMBER ===")
with pdfplumber.open(pdf_path) as pdf:
    print(pdf.pages[10].extract_text()[:500])

print("\n=== PYMUPDF (default) ===")
doc = pymupdf.open(pdf_path)
print(doc[10].get_text()[:500])

print("\n=== PYMUPDF (layout) ===")
# Note: layout might need special handling or just get_text("layout") if supported in this version
try:
    print(doc[10].get_text("layout")[:500])
except Exception as e:
    print("Layout error:", e)

doc.close()
