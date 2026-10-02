import pymupdf
doc = pymupdf.open("/Users/kartikeygupta/Desktop/Hackathons/sih_26108/backend/uploads/ef70ec28-66f1-4c99-9134-ef1d069c5877/hindi:english tender.pdf")
try:
    print(doc[10].get_text("layout"))
except Exception as e:
    import traceback
    traceback.print_exc()
