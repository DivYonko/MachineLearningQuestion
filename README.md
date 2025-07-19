Project Documentation: Enterprise Document Q&A System
1. Project Overview
	In today's data-driven enterprises, organizations are inundated with vast amounts of information stored across countless documents like reports, contracts, and internal wikis. Extracting specific, timely, and 		trustworthy information from this unstructured data is a significant challenge. Traditional search methods are often insufficient, and asking a standard Large Language Model (LLM) can lead to generic, non-				specific, or even fabricated answers (a phenomenon known as "hallucination").

	This project addresses this critical business problem by providing a sophisticated, multi-modal Question & Answering (Q&A) system. It is built on a Retrieval-Augmented Generation (RAG) architecture, a state 			of-	the-art technique that grounds the AI's responses in the actual content of an organization's private documents. This ensures that the answers are not only accurate and relevant but also verifiable.
	
	Users can upload various document types, including PDFs (.pdf), Microsoft Word documents (.docx), and plain text files (.txt). The system intelligently processes both the text and any embedded images, making 		all content searchable. It provides accurate, conversational answers complete with source citations, transforming how users interact with and leverage their internal knowledge base.

2.Key Features:

	Multi-Modal Ingestion: Processes and understands both text and images from various document formats, allowing users to ask questions like "What does the bar chart on page 5 show?".
	Retrieval-Augmented Generation (RAG): Provides answers grounded in the content of the uploaded documents. This is the core feature that ensures factual accuracy and relevance, as the AI is forced to use the 
	provided documents as its single source of truth.

	Source Citations: Every answer is accompanied by references to the source document and page number, building user trust and allowing for easy verification of the information.
	
 	Streaming Responses: Answers are streamed token-by-token for a responsive and modern user experience, providing immediate feedback to the user.
	Interactive UI: A user-friendly web interface built with Gradio allows for easy document upload and conversation, making the technology accessible to non-technical users.
	Session Management: Users can end a session to clear the document and its associated data from memory and storage, ensuring privacy and a clean slate for the next task.

3. System Architecture
The application is built on a classic RAG pipeline, a two-stage process that first prepares the data (Ingestion) and then uses it to answer questions (Inference).
	a) Ingestion Stage (Processing the Document)
			This is the preparatory stage where the system converts an unstructured document into a structured, searchable "knowledge base."
			Document Upload: The user uploads a file through the Gradio interface.
			Loading & Parsing: The SimpleDirectoryReader component from LlamaIndex automatically detects the file type (.pdf, .docx, etc.) and uses the appropriate parsing library (pypdf, docx2txt) to extract its 						contents. This process is multi-modal, meaning it intelligently separates text paragraphs, titles, and tables from embedded images.
			Embedding: This is the heart of the "understanding" process. The Gemini embedding model (models/embedding-001) is used to convert every piece of content (both text chunks and images) into a numerical 						representation called a vector. These vectors capture the semantic meaning of the content, allowing the system to understand concepts and relationships, not just keywords.
			Indexing & Storage: The generated vectors, along with their original content and metadata (e.g., file_name, page_label), are stored in a ChromaDB vector database. ChromaDB is a lightweight, open-source 					database designed specifically for this purpose and is ideal for local development. This process creates a persistent, searchable index on the disk, so a document only needs to be processed once.
	b) Inference Stage (Answering a Question)
			This stage is triggered every time the user asks a question.
			User Query: The user asks a question in the chat interface (e.g., "What were the key risks identified in the annual report?").
			Query Embedding: The same Gemini embedding model used during ingestion converts the user's question into a vector. This ensures that the question and the document content are represented in the same 							"semantic space."

			Retrieval: The system queries the ChromaDB database with the question's vector. It performs a similarity search to find the vectors from the document that are closest in meaning to the question's vector. 				The top k most relevant text and image chunks (nodes) are "retrieved." This is the critical step that narrows down the vast document to only the most relevant snippets of information.
			Context Augmentation: The retrieved text chunks and image data are compiled into a single block of information called the "context."
			Generation: This context is then passed to the Gemini 2.5 Flash language model, along with the original question and a carefully crafted prompt. The prompt instructs the model to act as a helpful assistant 			and formulate a comprehensive answer based only on the provided context. This constraint is what prevents the LLM from hallucinating or using outside knowledge. The model is also instructed to stream its 				response.
			Response & Citation: The streamed answer is displayed token-by-token in the UI. Once the stream is complete, the metadata from the source nodes retrieved in step 3 is formatted into citations and appended 				to the final answer.

4. Detailed Workflow
		Here is a more granular breakdown of the code's execution flow.
		Part A: The Ingestion Workflow (When a user uploads a document)
		Event Trigger: The uploader.upload() event in the Gradio UI is triggered when a file is dropped into the file component.
		Function Call: This event calls the process_document(file) function, passing the temporary file object provided by Gradio.
		Path & State Management: The function extracts the original filename and creates a unique path in the ./storage directory (e.g., ./storage/Annual_Report.pdf). This allows the system to cache the processed index and avoid re-processing the same file on subsequent uploads.
		Check for Existing Index: The code checks if this unique directory already exists. If it does, it loads the pre-built index directly from the disk using chromadb.PersistentClient, which is very fast.
		New Document Processing: If the index does not exist, the following occurs:
		SimpleDirectoryReader(input_files=[file.name]).load_data() is called. This is a key LlamaIndex component that inspects the file extension (.pdf, .docx, etc.) and uses the appropriate library (pypdf, docx2txt) to parse it. It automatically separates the content into TextNode and ImageNode objects.
		A ChromaVectorStore is initialized, pointing to the unique persistence directory for the document.
		VectorStoreIndex.from_documents() is the main workhorse. It iterates through all the TextNode and ImageNode objects. For each node, it calls the GeminiEmbedding model (models/embedding-001) to generate a vector embedding.
		These embeddings, along with the original content and metadata (file_name, page_label), are stored in the ChromaDB collection.
		Finally, index.storage_context.persist() saves the complete index to disk inside the ./storage/[filename] directory.
		Query Engine Creation: A query engine is created from the index using index.as_query_engine(). We configure it with streaming=True to enable the token-by-token response effect. This engine is stored in a global variable (query_engine) for the duration of the session.
		UI Feedback: The function returns a status message to the UI (e.g., "Ready to answer questions about Annual_Report.pdf") and makes the "End Current Session" button visible.
		Part B: The Inference Workflow (When a user asks a question)
		Event Trigger: The user types a message and hits Enter or clicks the "Submit" button. This triggers the .submit() or .click() event in the Gradio UI.
		Function Call: This event calls the chat_responder(message, history) generator function. A generator is used because we need to yield updates to the UI multiple times for a single response.
		Query Execution:
		The function first checks if the global query_engine has been initialized.
		query_engine.query(message) is called. Because the engine was created with streaming=True, this method is non-blocking and immediately returns a StreamingResponse object.
		Streaming Response Handling:
		An empty placeholder is added to the chat history for the bot's response.
		The code then iterates through the streaming_response.response_gen generator.
		In each iteration, a new token (a word or part of a word) is received from the Gemini LLM.
		This token is appended to the bot's message in the chat history.
		The yield keyword sends the updated chat history back to the Gradio UI, which re-renders the chatbot component. This process repeats rapidly, creating the "typing" effect.
		Citation Handling (Post-Streaming):
		Once the token generator is exhausted, the main answer has been fully streamed.
		The code then accesses streaming_response.source_nodes. This contains the list of text/image chunks that the retriever found most relevant to the user's question.
		It iterates through these nodes, extracts the file_name and page_label from their metadata, and adds them to a set to ensure there are no duplicate citations.
		The unique citations are formatted into a clean markdown string.
		This citation string is appended to the complete answer in the chat history.
		A final yield updates the UI one last time with the complete answer plus citations.

5. Technology Stack
		Orchestration: LlamaIndex - The core Python framework used for building and coordinating the entire RAG pipeline, from data loading to query execution.
		LLM & Embeddings: Google Gemini - The family of models providing the core AI capabilities. We use models/embedding-001 for creating semantic vectors and Gemini 2.5 Flash for its fast, high-quality, and multi-modal generation.
		Vector Database: ChromaDB - A lightweight, open-source vector store used to index and query the document embeddings efficiently. It's ideal for local development and can be swapped for a more scalable solution in production.
		Web UI: Gradio - A Python library used to rapidly create the interactive and user-friendly web application, making the backend accessible to end-users.
		Document Parsing: unstructured, pypdf, docx2txt - A suite of libraries that LlamaIndex uses under the hood to handle the extraction of text and image content from different file formats.
		Programming Language: Python 3.10 - The language used for the entire application.

6. Project Setup and Installation Guide
		Follow these steps to set up and run the project on a new machine.
		Prerequisites:
		Python 3.10 or higher.
		conda installed (recommended for managing environments).
		Step 1: Create and Activate a Virtual Environment It is highly recommended to create an isolated environment to avoid package conflicts with other projects on your system.
		# Create a new conda environment named 'rag_sprint'
		conda create -n rag_sprint python=3.10
		
		# Activate the environment
		conda activate rag_sprint
		
		Step 2: Obtain a Google API Key The application uses Gemini models, which require an API key for authentication.
		Go to Google AI Studio.
		Sign in and click "Get API key" > "Create API key".
		Copy the generated key.
		Step 3: Create the .env File This file securely stores your API key so it is not hard-coded into the application.
		In the root directory of your project, create a new file named .env.
		Open the file and add your API key in the following format:
		GOOGLE_API_KEY="YOUR_COPIED_API_KEY_HERE"
		
		
		Step 4: Install Dependencies The requirements.txt file contains all the necessary packages with their exact working versions to ensure reproducibility.
		Make sure you have the requirements.txt file in your project directory.
		Run the following command in your activated conda environment:
		pip install -r requirements.txt
		
		
		This will install all the required libraries automatically.
		Step 5: Run the Application Once the installation is complete, you can launch the Gradio web application.
		gradio app.py
		
		After running the command, your terminal will display a local URL (e.g., http://127.0.0.1:7860). Open this URL in your web browser to access the application.
		6. How to Use the Application
		Upload a Document: Drag and drop a .pdf, .docx, or .txt file into the "Upload Document" area on the left.
		Wait for Processing: The status box will show "Processing..." and then update to "Ready to answer questions about [your_file_name]."
		Ask a Question: Type your question into the chat box at the bottom and press Enter or click "Submit".
		View Response: The AI's answer will stream into the chat window, followed by the source citations.
		End Session: When you are finished with a document, click the "🗑️ End Current Session" button. This will delete the document's data from the backend and reset the interface, ready for a new document.

7. Potential Improvements & Future Work
			This project provides a strong foundation that can be extended with additional features:
			Support for More Formats: Add parsers for spreadsheets (.xlsx, .csv), presentations (.pptx), and URLs.
			Advanced Retrieval Strategies: Implement more complex retrieval logic, such as a hybrid search that combines vector similarity with traditional keyword search for better performance on specific queries (like product codes or names).
			User Authentication: For a true enterprise setting, add a login system to manage user access and secure documents.
			Conversation History: Enhance the prompt to include the last few turns of the conversation, allowing the AI to answer follow-up questions more effectively.
			Production Deployment: Containerize the application using Docker and deploy it on a cloud platform (AWS, GCP, Azure) for scalability and availability.

8. Conclusion
			The Enterprise Document Q&A System successfully demonstrates the power of Retrieval-Augmented Generation (RAG) in transforming unstructured data into an interactive and reliable knowledge source. By grounding its responses in user-provided documents and citing its sources, it provides a trustworthy alternative to standard LLMs for enterprise use cases. With a modular architecture and a clear path for future enhancements, this project serves as an excellent blueprint for building practical and valuable AI-powered tools.
			
