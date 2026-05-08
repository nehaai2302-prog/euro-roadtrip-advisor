import os
import streamlit as st
from dotenv import load_dotenv 
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load the .env file
load_dotenv()

# 1. Configuration - Change these to fine-tune your RAG
# This finds the absolute path to the folder where ingest.py is located
current_dir = Path(__file__).parent.absolute()

# Now we define our paths based on that directory
DATA_PATH = str(current_dir / "data")
CHROMA_PATH = str(current_dir / "chroma_db")
print(f"📂 Data will be loaded from: {DATA_PATH}")
print(f"🗄️ Database will be created at: {CHROMA_PATH}")

def run_ingestion():
        
    # Load your OpenAI Key (Ensure it's in your environment variables)
    api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment.")
        return

    print("🚀 Initializing EuroRoad Ingestion...")

    # STEP 1: LOAD DOCUMENTS
    # DirectoryLoader handles multiple files at once
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=lambda path: TextLoader(path, encoding="utf-8"))
    raw_documents = loader.load()
    print(f"📄 Successfully loaded {len(raw_documents)} documents.")

    # STEP 2: SPLIT INTO CHUNKS
    # We use 'Recursive' splitting to keep paragraphs together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # Approx 2-3 paragraphs
        chunk_overlap=150, # Keeps context between chunks
        add_start_index=True # Helpful for citations later
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"✂️ Created {len(chunks)} chunks from your knowledge base.")

    # STEP 3 & 4: EMBED AND STORE
    # This will create a folder named 'chroma_db' in your project
    print("🧠 Generating embeddings and saving to ChromaDB...")
    
    # We use OpenAI's latest 2026 embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # This line does the heavy lifting: embedding + saving
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print(f"✅ Ingestion Complete! Vector store saved at {CHROMA_PATH}")

if __name__ == "__main__":
    run_ingestion()