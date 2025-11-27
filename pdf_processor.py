import pdfplumber
import os # Achhi file handling ke liye 'os' library acchi hai

# Recommended: 'r' for Raw String ya Forward Slashes use karein
# A. Path Correction
pdf_file_path = r"C:\Users\MI\pro\history.pdf" 

extracted_data = [] # Saare pages ka data yahan store hoga

# Try-Except block mein poora process daalein (B. Repetition Fix)
try:
    # Check if file exists before trying to open
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"File not found at: {pdf_file_path}")

    # 1. File Load karna
    with pdfplumber.open(pdf_file_path) as pdf:
        print(f"File loaded successfully. Total pages found: {len(pdf.pages)}")
        
        # 2. Pages par loop karna
        for page_num, page in enumerate(pdf.pages):
            
            # 3. Text Extract karna
            text_content = page.extract_text()
            
            if text_content: 
                # 4. Data store karna (Metadata ke saath)
                extracted_data.append({
                    "page_number": page_num + 1, 
                    "raw_text": text_content,
                    # Note: source_file har chunk ke liye store karna RAG mein useful hai
                    "source_file": os.path.basename(pdf_file_path) 
                })
                print(f"Page {page_num + 1} extracted successfully.")

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    # Yeh general error handling hai (jaise PDF corrupted ho toh)
    print(f"An unexpected error occurred: {e}")

# Jab yeh code run ho jaayega, toh 'extracted_data' mein aapka clean data hoga.