import pymupdf
from pathlib import Path

pdf_path = Path("/Users/kartikeygupta/Desktop/Hackathons/sih_26108/backend/uploads/ef70ec28-66f1-4c99-9134-ef1d069c5877/hindi:english tender.pdf")

doc = pymupdf.open(pdf_path)
print("=== PYMUPDF blocks ===")
for block in doc[10].get_text("blocks"):
    print(block[4])
doc.close()
