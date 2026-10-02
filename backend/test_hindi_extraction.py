import time
from pathlib import Path

pdf_path = Path("/Users/kartikeygupta/Desktop/Hackathons/sih_26108/backend/uploads/ef70ec28-66f1-4c99-9134-ef1d069c5877/hindi:english tender.pdf")

print("--- Testing PyMuPDF ---")
try:
    import fitz
    t0 = time.time()
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Chars extracted: {len(text)}")
    print("Sample (Page 2):")
    print(doc[1].get_text()[:300])
except Exception as e:
    print(f"Error: {e}")

print("\n--- Testing pypdfium2 ---")
try:
    import pypdfium2 as pdfium
    t0 = time.time()
    doc2 = pdfium.PdfDocument(pdf_path)
    text2 = "\n".join([page.get_textpage().get_text_range() for page in doc2])
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Chars extracted: {len(text2)}")
    print("Sample (Page 2):")
    print(doc2[1].get_textpage().get_text_range()[:300])
except Exception as e:
    print(f"Error: {e}")
