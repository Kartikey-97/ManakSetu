import pymupdf4llm
from pathlib import Path

pdf_path = Path("/Users/kartikeygupta/Desktop/Hackathons/sih_26108/backend/uploads/ef70ec28-66f1-4c99-9134-ef1d069c5877/hindi:english tender.pdf")

try:
    md_text = pymupdf4llm.to_markdown(str(pdf_path), pages=[10])
    print("=== PYMUPDF4LLM Markdown Output (Page 10) ===")
    print(md_text[:1000])
except Exception as e:
    print(f"Error: {e}")
