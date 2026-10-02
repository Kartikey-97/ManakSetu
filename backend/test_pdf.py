import time
from pathlib import Path
import pdfplumber
import pypdfium2

pdf_path = "/Users/kartikeygupta/Desktop/Hackathons/sih_26108/backend/uploads/ef70ec28-66f1-4c99-9134-ef1d069c5877/hindi:english tender.pdf"

if Path(pdf_path).exists():
    print("Testing pypdfium2...")
    t0 = time.time()
    pdf = pypdfium2.PdfDocument(pdf_path)
    text2 = "\n".join([page.get_textpage().get_text_range() for page in pdf])
    t1 = time.time()
    print(f"pypdfium2 took: {t1-t0:.2f}s, extracted {len(text2)} chars")

    print("\nTesting pdfplumber...")
    t0 = time.time()
    with pdfplumber.open(pdf_path) as pdf_plumb:
        text = "\n".join([page.extract_text(x_tolerance=3, y_tolerance=3) or "" for page in pdf_plumb.pages])
    t1 = time.time()
    print(f"pdfplumber took: {t1-t0:.2f}s, extracted {len(text)} chars")
