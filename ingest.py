import os
import sys
from dotenv import load_dotenv
import logging

# LlamaIndex libraries - UPDATED for Gemini
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
# This now comes from a separate package for Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
# This now comes from a separate package for Gemini
from llama_index.llms.gemini import Gemini

# ChromaDB library
import chromadb

# --- Configuration ---
# Set up logging to see what's happening
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Load API Key from .env file
load_dotenv()
# UPDATED to use GOOGLE_API_KEY
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the .env file.")

# --- Constants ---
# Define the directories for data and storage
DATA_DIR = "./data"
STORAGE_DIR = "./storage"
PERSIST_DIR = os.path.join(STORAGE_DIR, "chroma")

# --- Main Ingestion Function ---
def ingest_documents(data_path, persist_path):
    """
    Processes a PDF document from `data_path`, extracts text and images,
    creates a multi-modal vector index using Gemini, and persists it.
    """
    print(f"Starting ingestion from '{data_path}'...")

    # Check if the data directory exists
    if not os.path.exists(data_path):
        print(f"Error: Data directory '{data_path}' not found.")
        os.makedirs(data_path)
        print("Created the './data' directory. Please add a PDF file to it and run again.")
        return None

    if not os.listdir(data_path):
        print(f"The '{data_path}' directory is empty. Please add a PDF to process.")
        return None

    print("Loading documents from the data directory...")
    documents = SimpleDirectoryReader(data_path).load_data()
    
    print(f"Successfully loaded {len(documents)} document(s).")

    # Initialize the Gemini multi-modal LLM
    # Gemini can handle both text and image inputs in a single model
    gemini_llm = Gemini(
        model_name="models/gemini-pro-vision", 
        api_key=google_api_key
    )

    # Initialize the Gemini embedding model
    gemini_embedding = GeminiEmbedding(
        model_name="models/embedding-001",
        api_key=google_api_key
    )

    # Check if a database already exists
    if os.path.exists(persist_path):
        print(f"Found existing database at '{persist_path}'. Loading...")
        db = chromadb.PersistentClient(path=persist_path)
        chroma_collection = db.get_or_create_collection("multimodal_rag_gemini")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Load the index from storage
        index = VectorStoreIndex.from_vector_store(
            vector_store, 
            embed_model=gemini_embedding
        )
        print("Existing index loaded.")
    else:
        print("No existing database found. Creating a new one.")
        db = chromadb.PersistentClient(path=persist_path)
        chroma_collection = db.get_or_create_collection("multimodal_rag_gemini")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create the Vector Store Index using Gemini models
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=gemini_embedding,
            show_progress=True,
        )
        
        index.storage_context.persist(persist_dir=persist_path)
        print(f"New index created and persisted to '{persist_path}'.")

    print("\n--- Ingestion Complete ---")
    return index

# --- Script Execution ---
if __name__ == "__main__":
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)
        
    ingest_documents(DATA_DIR, PERSIST_DIR)
    print(f"\nYou can now query your document. The indexed data is stored in '{PERSIST_DIR}'.")
