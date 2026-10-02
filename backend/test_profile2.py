from pathlib import Path
from kartikey.document_processing.extractor import extract_text

pdf_path = Path("/Users/kartikeygupta/Desktop/Hackathons/sih_26108/backend/uploads/ef70ec28-66f1-4c99-9134-ef1d069c5877/hindi:english tender.pdf")
text = extract_text(pdf_path)
print(f"Extracted {len(text) if text else 0} characters.")
print(text[:500] if text else "None")
