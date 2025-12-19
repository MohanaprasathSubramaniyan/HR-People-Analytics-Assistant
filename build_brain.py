import os
import glob
from langchain_community.document_loaders import PyPDFLoader
# UPDATED IMPORT: Using the new specific library
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# CONFIGURATION
DATA_FOLDER = r"C:\HR_Project"
DB_PATH = os.path.join(DATA_FOLDER, "chroma_db")

def build_knowledge_base():
    print("🚀 Starting the Brain Builder...")
    
    # 1. FIND PDFS
    pdf_files = glob.glob(os.path.join(DATA_FOLDER, "*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found! Did you copy them to C:\\HR_Project?")
        return

    # 2. LOAD PDFS
    all_documents = []
    print(f"📂 Found {len(pdf_files)} PDF files. Reading them now...")
    
    for file_path in pdf_files:
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_documents.extend(docs)
            print(f"   - Read: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"   - Error: {e}")

    # 3. SPLIT TEXT
    print("\n✂️  Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_documents)

    # 4. SAVE TO DATABASE
    print("\n🧠 Creating the Brain (chroma_db folder)...")
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_function,
        persist_directory=DB_PATH
    )
    
    print(f"🎉 Success! Created 'chroma_db' in {DATA_FOLDER}")

if __name__ == "__main__":
    build_knowledge_base()