import time
import pdfplumber
from pathlib import Path

pdf_path = Path("/Users/kartikeygupta/Desktop/Hackathons/sih_26108/backend/uploads/ef70ec28-66f1-4c99-9134-ef1d069c5877/hindi:english tender.pdf")

t0 = time.time()
with pdfplumber.open(pdf_path) as pdf:
    pages = []
    for page in pdf.pages:
        pages.append(page.extract_text())
t1 = time.time()
print(f"Default Time: {t1-t0:.2f}s")

t0 = time.time()
with pdfplumber.open(pdf_path) as pdf:
    pages = []
    for page in pdf.pages:
        pages.append(page.extract_text(x_tolerance=3, y_tolerance=3))
t1 = time.time()
print(f"Tolerance Time: {t1-t0:.2f}s")
