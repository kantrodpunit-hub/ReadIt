import os
import pdfplumber
from google import genai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# LangChain Components (Fixed and Stable Imports)
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter 

# --- Configuration ---
pdf_file_path = "history.pdf" 
CHUNK_SIZE = 1000  
CHUNK_OVERLAP = 150 

# 🛑 FINAL FIX: API Key ko seedhe yahan define karein 
# Please replace "YOUR_GENERATED_API_KEY_HERE" with your actual key.
# For example: GEMINI_KEY = "AIzaSyAqZIGfp7v2mSU5Mj9-ZNcy6B_-pdq34nE"
GEMINI_KEY = "AIzaSyAqZIGfp7v2mSU5Mj9-ZNcy6B_-pdq34nE"
# ---------------------

# API Key Check (Zaroori)
if GEMINI_KEY == "YOUR_GENERATED_API_KEY_HERE" or not GEMINI_KEY:
    print("🛑 Error: Please enter your actual Gemini API key in the GEMINI_KEY variable.")
    exit()

# ===============================================
# PART 1 & 2: PDF Parsing and Chunking
# ===============================================
print("1. Starting PDF Parsing and Chunking...")
extracted_pages_data = []
final_chunks = []

try:
    with pdfplumber.open(pdf_file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text_content = page.extract_text()
            if text_content: 
                extracted_pages_data.append({"page_number": page_num + 1, "raw_text": text_content})
                
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""] 
    )
    
    for item in extracted_pages_data:
        chunks = text_splitter.split_text(item["raw_text"])
        final_chunks.extend(chunks)
    
    print(f"✅ Chunking complete. Total chunks created: {len(final_chunks)}")

except Exception as e:
    print(f"🛑 Error during data prep: {e}")
    exit()

# ===============================================
# PART 3: Vectorization aur FAISS Storage
# ===============================================
print("2. Creating Embeddings and Storing in FAISS (Using Gemini API)...")

# 1. Embedding Model (Key pass ki gayi hai)
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004", 
    google_api_key=GEMINI_KEY
)

# 2. Vector Store Banana
try:
    vectorstore = FAISS.from_texts(
        texts=final_chunks,
        embedding=embeddings_model
    )
    print("✅ FAISS Vector Store successfully created with Gemini Embeddings.")
except Exception as e:
    print(f"🛑 Fatal Error during Vectorization (Gemini API Call): {e}")
    print("Please check your API key or API quota.")
    exit()

# ===============================================
# PART 4: RAG Chain Banana (RetrievalQA Bypassing)
# ===============================================
print("3. Setting up Custom RAG Chain...")

# 1. Retrieval Setup
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. Generation Setup (Gemini LLM)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.1, 
    google_api_key=GEMINI_KEY
)

# 3. Prompt Template
template = """You are an expert tutor. Answer the user's question only based on the 
provided context from the textbook. Do not use external knowledge. 
Context: {context}
Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# 4. RAG Chain ki definition (Low-level, stable approach)
def format_docs(docs):
    # Documents ko string format mein jodta hai
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    # Context ko retrieve karke format karna
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    # Prompt, LLM, aur Output Parser ko chain karna
    | prompt
    | llm
    | StrOutputParser()
)

# ===============================================
# FINAL STEP: Question Puchhna
# ===============================================
print("\n--- RAG System Ready ---")

while True:
    query = input("Ask your question about the history book (Type 'exit' to quit): \n> ")
    if query.lower() == 'exit':
        break
    
    print("\n[Thinking...]")
    
    try:
        # Chain ko call karna (Search -> Context -> Gemini Answer)
        response = rag_chain.invoke(query)
        
        print("\n==============================================")
        print("✅ BOOK TUTOR ANSWER:")
        print(response)
        print("==============================================")
    except Exception as e:
        print(f"🛑 Error generating answer: {e}")

print("Application closed.")