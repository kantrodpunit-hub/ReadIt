import pdfplumber
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
# 1. Apni PDF file ka naam yahan daalein (Agar file same folder mein hai)
pdf_file_path = "history.pdf" 
# Agar file ka path C:\Users\... jaisa hai, toh 'r' use karein:
# pdf_file_path = r"C:\Users\MI\pro\history.pdf" 

# 2. Chunking Settings (Inhe aap adjust kar sakte hain)
CHUNK_SIZE = 1000  # Har tukde mein characters ki maximum sankhya
CHUNK_OVERLAP = 150 # Chunks ke beech mein kitne characters common honge
# ---------------------

# Final chunks is list mein store honge
final_chunks = []
extracted_pages_data = []

# ===============================================
# PART 1: PDF Parsing aur Text Extraction
# ===============================================

print(f"Starting PDF parsing for: {pdf_file_path}")

try:
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"Error: File not found at: {pdf_file_path}")

    with pdfplumber.open(pdf_file_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"File loaded successfully. Total pages found: {total_pages}")
        
        for page_num, page in enumerate(pdf.pages):
            # Text Extract karna
            text_content = page.extract_text()
            
            if text_content: 
                # Data ko structured format mein store karna
                extracted_pages_data.append({
                    "page_number": page_num + 1, 
                    "raw_text": text_content,
                    "source_file": os.path.basename(pdf_file_path) 
                })
        
        print(f"✅ {len(extracted_pages_data)} pages successfully extracted and prepared for chunking.")

except FileNotFoundError as e:
    print(f"🛑 Error: {e}. Please check the file path.")
    exit()
except Exception as e:
    print(f"🛑 An unexpected error occurred during PDF reading: {e}")
    exit()


# ===============================================
# PART 2: Text Chunking (Splitter ka use karke)
# ===============================================

# Splitter object banana
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""] 
)

print(f"Starting chunking with Size={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP}...")

for item in extracted_pages_data:
    raw_text = item["raw_text"]
    
    # Text ko chunks mein split karna
    chunks = text_splitter.split_text(raw_text)
    
    # Har chunk ke saath original metadata jodna
    for chunk in chunks:
        final_chunks.append({
            "text_content": chunk,
            # Metadata: Page Number aur Source File ko har chunk ke saath jodna
            "metadata": {
                "page_number": item["page_number"],
                "source_file": item["source_file"]
            }
        })

print(f"✅ Chunking complete!")
print(f"Total pages extracted: {len(extracted_pages_data)}")
print(f"Total chunks created: {len(final_chunks)}")
print("---")
print("First chunk example:")
print(final_chunks[0]['text_content'][:200] + "...") # Pehle chunk ke 200 characters dikhayein

# ===============================================
# FINAL STEP: Data ko Verify karna
# ===============================================

# Ab 'final_chunks' list mein aapka data ready hai Vectorization ke liye!
# Har item ek dictionary hai jismein 'text_content' aur 'metadata' hai.