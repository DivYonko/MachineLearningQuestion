import os
import sys
import logging
import shutil
from dotenv import load_dotenv
import gradio as gr

# LlamaIndex libraries
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
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
    raise ValueError("GOOGLE_API_KEY is not set in the .env file or is invalid.")

# --- Constants ---
# Define directories
STORAGE_DIR = "./storage"

# --- Global State Management ---
# This will hold the query engine for the currently loaded document
query_engine = None
current_file_name = None

# --- Core Functions ---

def initialize_llms():
    """Initializes and returns Gemini models."""
    embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=google_api_key)
    llm = Gemini(model_name="models/gemini-1.5-flash-latest", api_key=google_api_key)
    return embed_model, llm

def process_document(file):
    """
    Processes a single uploaded document and builds a query engine for it.
    """
    global query_engine, current_file_name

    if file is None:
        return "*Status: No file uploaded. Please upload a document.*", gr.update(visible=False)

    file_name = os.path.basename(file.name)
    print(f"Processing document: {file_name}")

    # Create a unique storage path for this document's index
    persist_dir = os.path.join(STORAGE_DIR, file_name)
    
    embed_model, llm = initialize_llms()

    # Load from storage if index already exists
    if os.path.exists(persist_dir):
        print("Loading existing index from storage...")
        db = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = db.get_or_create_collection("multimodal_rag_gemini")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    else:
        print("Creating new index...")
        # The uploaded file is temporary, so we read it directly
        # SimpleDirectoryReader will automatically handle PDF, DOCX, and TXT
        documents = SimpleDirectoryReader(input_files=[file.name]).load_data()
        
        db = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = db.get_or_create_collection("multimodal_rag_gemini")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, embed_model=embed_model, show_progress=True
        )
        index.storage_context.persist(persist_dir=persist_dir)

    # Create a query engine that supports streaming
    query_engine = index.as_query_engine(
        llm=llm, 
        similarity_top_k=5, 
        response_mode="compact",
        streaming=True # Enable streaming
    )
    current_file_name = file_name
    
    print("Document processed successfully.")
    # Return updates for the UI
    return f"Ready to answer questions about **{file_name}**.", gr.update(visible=True)

def chat_responder(message, history):
    """
    Responds to a user message, streaming the response and adding citations.
    """
    if not message.strip():
        yield "", history
        return
        
    if query_engine is None:
        history.append((message, "Error: Please upload and process a document first."))
        yield "", history
        return

    print(f"Querying with message: '{message}'")
    
    # --- FIX: Use .query() which returns a StreamingResponse object ---
    streaming_response = query_engine.query(message)
    
    # Append user message and an empty placeholder for the bot's response
    history.append((message, ""))
    
    # Stream the response tokens
    for token in streaming_response.response_gen:
        history[-1] = (message, history[-1][1] + token)
        yield "", history
        
    # --- Process source nodes after streaming is complete ---
    source_nodes = streaming_response.source_nodes
    citations = set() # Use a set to store unique citations
    
    if source_nodes:
        for node in source_nodes:
            file_name = node.metadata.get('file_name', 'N/A')
            page_label = node.metadata.get('page_label', 'N/A')
            citations.add(f"'{file_name}', Page {page_label}")
            
    # Append the formatted citations to the final response
    if citations:
        formatted_citations = "\n\n*Sources:*\n" + "\n".join([f"- {c}" for c in sorted(list(citations))])
        history[-1] = (message, history[-1][1] + formatted_citations)
        yield "", history

def end_session():
    """
    Clears the session by deleting the index and resetting the state.
    """
    global query_engine, current_file_name
    
    if current_file_name:
        print(f"Ending session for {current_file_name}. Deleting index.")
        persist_dir = os.path.join(STORAGE_DIR, current_file_name)
        if os.path.exists(persist_dir):
            try:
                shutil.rmtree(persist_dir)
                print("Successfully deleted index directory.")
            except Exception as e:
                print(f"Error deleting directory {persist_dir}: {e}")

    # Reset global state
    query_engine = None
    current_file_name = None
    
    # Return updates to reset the UI components
    return (
        None,  # Clear the file uploader
        [],    # Clear the chatbot history
        "*Status: Session ended. Ready for new document.*", # Update status
        gr.update(visible=False) # Hide the end session button
    )

# --- Gradio UI ---

# Custom theme for a more modern look
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
).set(
    body_background_fill="#f0f4f8",
    block_background_fill="white",
    block_border_width="1px",
    block_shadow="*shadow_sm",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_secondary_background_fill="*neutral_100",
    button_secondary_background_fill_hover="*neutral_200",
)

with gr.Blocks(theme=theme, title="Enterprise Q&A") as demo:
    gr.Markdown(
        """
        <div style="text-align: center; padding: 20px 0;">
            <h1>📄 Enterprise Document Q&A</h1>
            <p style="color: #555;">Upload a PDF, DOCX, or TXT file and ask questions. The system uses Gemini to understand text and images.</p>
        </div>
        """
    )
    
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=350):
            # Replaced gr.Box() with gr.Group() for wider compatibility
            with gr.Group():
                gr.Markdown("### ⚙️ Control Panel")
                # Updated to accept more file types
                uploader = gr.File(label="Upload Document", file_types=[".pdf", ".docx", ".txt"], type="filepath")
                status_box = gr.Markdown(value="*Status: Waiting for document...*")
                end_session_btn = gr.Button("🗑️ End Current Session", visible=False)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=600, bubble_full_width=False, avatar_images=("https://placehold.co/40x40/000000/FFFFFF?text=U", "https://placehold.co/40x40/007bff/FFFFFF?text=B"))
            with gr.Row():
                message_box = gr.Textbox(
                    placeholder="Ask a question about the document...",
                    show_label=False,
                    autofocus=True,
                    scale=4
                )
                submit_btn = gr.Button("Submit", variant="primary", scale=1, min_width=150)

    # --- Event Listeners ---

    # When a file is uploaded, process it
    uploader.upload(
        fn=process_document, 
        inputs=[uploader], 
        outputs=[status_box, end_session_btn], 
        queue=True
    )

    # Handle chat submission via button click
    submit_btn.click(
        fn=chat_responder,
        inputs=[message_box, chatbot],
        outputs=[message_box, chatbot]
    )
    
    # Handle chat submission via Enter key
    message_box.submit(
        fn=chat_responder,
        inputs=[message_box, chatbot],
        outputs=[message_box, chatbot]
    )

    # Handle the end session button click
    end_session_btn.click(
        fn=end_session,
        inputs=[],
        outputs=[uploader, chatbot, status_box, end_session_btn],
        queue=True
    )

# --- Main Execution ---

if __name__ == "__main__":
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)
        
    print("Launching Gradio App...")
    demo.launch(share=True)
