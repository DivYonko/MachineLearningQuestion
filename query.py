import os
import sys
from dotenv import load_dotenv
import logging

# LlamaIndex libraries
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

# ChromaDB library
import chromadb

# --- Configuration ---
# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Load API Key from .env file
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the .env file.")

# --- Constants ---
# Directory where the index is stored
STORAGE_DIR = "./storage"
PERSIST_DIR = os.path.join(STORAGE_DIR, "chroma")

# --- Main Query Function ---
def query_documents():
    """
    Loads the persisted index and allows the user to ask questions about the documents.
    """
    # Check if the storage directory exists
    if not os.path.exists(PERSIST_DIR):
        print(f"Error: Storage directory '{PERSIST_DIR}' not found. Please run ingest.py first.")
        return

    print(f"Loading index from '{PERSIST_DIR}'...")

    # Initialize the Gemini embedding model (must match the one used for ingestion)
    embed_model = GeminiEmbedding(
        model_name="models/embedding-001",
        api_key=google_api_key
    )

    # Initialize the Gemini LLM
    llm = Gemini(
        model_name="models/gemini-2.5-flash", # Using a fast and capable model for Q&A
        api_key=google_api_key
    )

    # Load the persisted ChromaDB vector store
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_or_create_collection("multimodal_rag_gemini")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Load the index from the vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # Create a query engine from the index
    # We can configure it to retrieve more context if needed
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=3, # Retrieve top 3 most similar nodes
        response_mode="compact"
    )

    print("\n--- Query Engine Ready ---")
    print("Ask a question about your document. Type 'exit' or 'quit' to stop.")

    # Start a loop to ask questions
    while True:
        try:
            query_text = input("\nQuestion: ")
            if query_text.lower() in ['exit', 'quit']:
                break
            
            print("Querying...")
            response = query_engine.query(query_text)
            
            print("\nResponse:")
            print(str(response))

            # Optional: Print the source nodes used for the response
            # print("\nSource Nodes:")
            # for node in response.source_nodes:
            #     print(f"- Node ID: {node.node_id}, Score: {node.score:.4f}")
            #     print(f"  Content: {node.text[:150]}...") # Print snippet of the source

        except EOFError:
            # This handles Ctrl+D to exit
            break
        except KeyboardInterrupt:
            # This handles Ctrl+C to exit
            print("\nExiting...")
            break

# --- Script Execution ---
if __name__ == "__main__":
    query_documents()