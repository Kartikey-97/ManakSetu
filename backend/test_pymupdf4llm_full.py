import time
import pymupdf4llm
from pathlib import Path

pdf_path = Path("/Users/kartikeygupta/Desktop/Hackathons/sih_26108/backend/uploads/ef70ec28-66f1-4c99-9134-ef1d069c5877/hindi:english tender.pdf")

t0 = time.time()
md_text = pymupdf4llm.to_markdown(str(pdf_path))
t1 = time.time()
print(f"Time: {t1-t0:.2f}s")
print(f"Chars extracted: {len(md_text)}")
